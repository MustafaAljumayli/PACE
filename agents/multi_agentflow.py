"""
Multi-Agent Turn Loop: Planner → Executor → Verifier → Generator.

Modeled after AgentFlow's 4-module architecture but simplified for
PACE benchmarking. Each module is a separate LLM call with a specialized
system prompt.

Turn protocol (one "turn" = one full cycle through all 4 modules):
  1. PLANNER:   Given query + memory, decide next sub-goal + tool to use
  2. EXECUTOR:  Execute the planned tool call
  3. VERIFIER:  Check if the result is useful / correct / sufficient
  4. GENERATOR: Produce updated answer given all accumulated information

The key difference from SingleReActAgent: each turn involves 4 LLM calls
(or 3 if using the same model). This is more expensive per-turn but
potentially more accurate, making PACE's cost savings more impactful.
"""

from __future__ import annotations

import json
from typing import Any

from openai import OpenAI

from agents import BaseAgent
from pace.trajectory import Trajectory, TurnState


# ─── Module System Prompts ───────────────────────────────────────────

PLANNER_PROMPT = """\
You are the PLANNER in a multi-agent reasoning system. Your job is to decide
the next action to take toward answering the user's question.

Available tools: {tool_descriptions}

Given the question and the memory of previous steps, output a JSON plan:
{{
    "sub_goal": "What specific information do we need next?",
    "tool": "which_tool_to_call",
    "tool_input": "what to pass to the tool",
    "reasoning": "Why this is the best next step"
}}

Memory of previous steps:
{memory}
"""

VERIFIER_PROMPT = """\
You are the VERIFIER in a multi-agent reasoning system. Your job is to assess
whether the latest tool result is useful and whether we have enough information
to answer the question.

Question: {question}
Latest tool output: {tool_output}
Accumulated evidence: {memory}

Output JSON:
{{
    "is_useful": true/false,
    "is_sufficient": true/false,
    "issues": "Any problems with the result",
    "suggestion": "What to do next if not sufficient"
}}
"""

GENERATOR_PROMPT = """\
You are the GENERATOR in a multi-agent reasoning system. Your job is to produce
the best possible answer to the user's question given all accumulated evidence.

Question: {question}

All accumulated evidence:
{memory}

Provide a clear, accurate answer based solely on the evidence above.
"""


def _build_memory(trajectory: Trajectory) -> str:
    """Build a text summary of all previous turns for module prompts."""
    if not trajectory.turns:
        return "(No previous steps)"
    lines = []
    for t in trajectory.turns:
        lines.append(
            f"Turn {t.turn_number}: "
            f"Tool={t.tool_called or 'none'} | "
            f"Answer={t.answer[:200]}"
        )
        if t.retrieved_context:
            lines.append(f"  Evidence: {t.retrieved_context[:300]}")
    return "\n".join(lines)


def _extract_top_logprobs(choice: Any) -> list[list[float]]:
    """Extract per-token top-k logprobs from a chat completion choice."""
    content = getattr(getattr(choice, "logprobs", None), "content", None)
    if not content:
        return []

    result: list[list[float]] = []
    for tok in content:
        top = getattr(tok, "top_logprobs", None) or []
        vals = [getattr(item, "logprob", None) for item in top]
        vals = [float(v) for v in vals if v is not None]
        if not vals:
            chosen = getattr(tok, "logprob", None)
            if chosen is not None:
                vals = [float(chosen)]
        if vals:
            result.append(vals)
    return result


class MultiAgentTeam(BaseAgent):
    """
    4-module multi-agent team: Planner → Executor → Verifier → Generator.

    This is your Condition B: multi-agent team, multi-turn.

    Each "turn" runs the full 4-module pipeline. PACE monitors at the
    turn level (after Generator produces an answer), not at the
    individual module level.
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        max_turns: int = 10,
        tools: dict[str, Any] | None = None,
        planner_model: str | None = None,
        verifier_model: str | None = None,
        generator_model: str | None = None,
    ):
        super().__init__(model=model, max_turns=max_turns, tools=tools)
        self.planner_model = planner_model or model
        self.verifier_model = verifier_model or model
        self.generator_model = generator_model or model
        self.client = OpenAI()

    def _llm_call(
        self,
        model: str,
        messages: list[dict],
        json_mode: bool = True,
        capture_logprobs: bool = False,
    ) -> tuple[str, int, list[list[float]]]:
        """Make an LLM call. Returns (response_text, token_count, top_logprobs)."""
        kwargs = {"model": model, "messages": messages, "temperature": 0.3, "max_tokens": 1500}
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        if capture_logprobs:
            kwargs["logprobs"] = True
            kwargs["top_logprobs"] = 20

        try:
            resp = self.client.chat.completions.create(**kwargs)
        except Exception:
            if capture_logprobs:
                kwargs.pop("logprobs", None)
                kwargs.pop("top_logprobs", None)
                resp = self.client.chat.completions.create(**kwargs)
            else:
                raise

        text = resp.choices[0].message.content or ""
        tokens = resp.usage.total_tokens if resp.usage else 0
        top_logprobs = _extract_top_logprobs(resp.choices[0]) if capture_logprobs else []
        return text, tokens, top_logprobs

    def _execute_turn(
        self,
        query: str,
        trajectory: Trajectory,
        turn_number: int,
    ) -> TurnState:
        memory = _build_memory(trajectory)
        tool_descs = "\n".join(
            f"- {name}: {tool.get('description', 'No description')}"
            for name, tool in self.tools.items()
        )
        total_tokens = 0

        # ── STEP 1: PLANNER ──────────────────────────────────────
        planner_prompt = PLANNER_PROMPT.format(
            tool_descriptions=tool_descs, memory=memory
        )
        plan_raw, tokens, _ = self._llm_call(
            self.planner_model,
            [
                {"role": "system", "content": planner_prompt},
                {"role": "user", "content": f"Question: {query}\nThis is planning step {turn_number}."},
            ],
        )
        total_tokens += tokens

        try:
            plan = json.loads(plan_raw)
        except json.JSONDecodeError:
            plan = {"sub_goal": "", "tool": "none", "tool_input": "", "reasoning": plan_raw}

        tool_name = plan.get("tool", "none")
        tool_input = plan.get("tool_input", "")
        reasoning = plan.get("reasoning", "")

        # ── STEP 2: EXECUTOR ─────────────────────────────────────
        retrieved = ""
        if tool_name and tool_name != "none" and tool_name in self.tools:
            tool_fn = self.tools[tool_name].get("function")
            if tool_fn:
                try:
                    retrieved = tool_fn(tool_input)
                except Exception as e:
                    retrieved = f"Tool error: {e}"

        # ── STEP 3: VERIFIER ─────────────────────────────────────
        verifier_prompt = VERIFIER_PROMPT.format(
            question=query,
            tool_output=retrieved[:2000] if retrieved else "(no tool output)",
            memory=memory,
        )
        verify_raw, tokens, _ = self._llm_call(
            self.verifier_model,
            [
                {"role": "system", "content": verifier_prompt},
                {"role": "user", "content": "Assess the current state."},
            ],
        )
        total_tokens += tokens

        try:
            verification = json.loads(verify_raw)
        except json.JSONDecodeError:
            verification = {"is_useful": True, "is_sufficient": False}

        # Append verification notes to reasoning
        reasoning += f"\n[Verifier] useful={verification.get('is_useful')}, sufficient={verification.get('is_sufficient')}"

        # ── STEP 4: GENERATOR ────────────────────────────────────
        # Accumulate all evidence for the generator
        all_evidence = memory
        if retrieved:
            all_evidence += f"\n\nLatest evidence (turn {turn_number}):\n{retrieved[:2000]}"

        gen_prompt = GENERATOR_PROMPT.format(question=query, memory=all_evidence)
        answer_raw, tokens, token_top_logprobs = self._llm_call(
            self.generator_model,
            [
                {"role": "system", "content": gen_prompt},
                {"role": "user", "content": "Produce the best answer."},
            ],
            json_mode=False,  # Generator outputs free text
            capture_logprobs=True,
        )
        total_tokens += tokens

        return TurnState(
            turn_number=turn_number,
            answer=answer_raw.strip(),
            retrieved_context=retrieved,
            tool_called=tool_name if tool_name != "none" else "",
            reasoning_text=reasoning,
            token_count=total_tokens,
            metadata={
                "plan": plan,
                "verification": verification,
                "token_top_logprobs": token_top_logprobs,
            },
        )
