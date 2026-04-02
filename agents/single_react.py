"""
Single-Agent ReAct Loop.

One LLM does everything: reason about the query, decide which tool to call,
interpret tool output, update its answer. This is the standard baseline.

Turn protocol:
  1. Build prompt with query + conversation history + available tools
  2. LLM generates thought + action (tool call or final answer)
  3. If tool call: execute tool, get observation
  4. LLM generates updated answer given observation
  5. Return TurnState
"""

from __future__ import annotations

import json
from typing import Any

from openai import OpenAI

from agents import BaseAgent
from pace.trajectory import Trajectory, TurnState


REACT_SYSTEM_PROMPT = """\
You are a research assistant that solves complex questions through iterative \
reasoning and tool use. You have access to the following tools:

{tool_descriptions}

For each turn, you will:
1. THINK: Analyze what you know so far and what you still need.
2. ACT: Call exactly one tool to gather new information, OR state your final answer.
3. ANSWER: Based on all information so far, provide your current best answer.

Respond in this exact JSON format:
{{
    "thought": "Your reasoning about what to do next...",
    "action": {{
        "tool": "tool_name_or_none",
        "input": "tool input string"
    }},
    "current_answer": "Your best answer given everything so far"
}}

If you are confident in your answer and don't need more information, set tool to "none".
"""


def _build_history(trajectory: Trajectory) -> list[dict]:
    """Build conversation history from trajectory for the LLM."""
    messages = []
    for turn in trajectory.turns:
        # Agent's previous reasoning
        messages.append({
            "role": "assistant",
            "content": json.dumps({
                "thought": turn.reasoning_text[:500],
                "action": {"tool": turn.tool_called, "input": "..."},
                "current_answer": turn.answer[:500],
            }),
        })
        # Tool observation
        if turn.retrieved_context:
            messages.append({
                "role": "user",
                "content": f"[Tool Output from {turn.tool_called}]:\n{turn.retrieved_context[:2000]}",
            })
    return messages


def _extract_top_logprobs(choice: Any) -> list[list[float]]:
    """
    Extract per-token top logprobs from OpenAI Chat Completions choice.

    Returns:
        A list where each item corresponds to one generated token and contains
        the token's top-k logprobs.
    """
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


class SingleReActAgent(BaseAgent):
    """
    Single-agent ReAct: one LLM reasons + acts + answers.

    This is your Condition A: single agent, multi-turn.
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        max_turns: int = 10,
        tools: dict[str, Any] | None = None,
    ):
        super().__init__(model=model, max_turns=max_turns, tools=tools)
        self.client = OpenAI()

    def _execute_turn(
        self,
        query: str,
        trajectory: Trajectory,
        turn_number: int,
    ) -> TurnState:
        # Build tool descriptions
        tool_descs = "\n".join(
            f"- {name}: {tool.get('description', 'No description')}"
            for name, tool in self.tools.items()
        )

        system = REACT_SYSTEM_PROMPT.format(tool_descriptions=tool_descs)

        # Build messages
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": f"Question: {query}"},
        ]
        messages.extend(_build_history(trajectory))

        if turn_number > 1:
            messages.append({
                "role": "user",
                "content": (
                    f"This is turn {turn_number}. Continue reasoning. "
                    "Call a tool if you need more information, or refine your answer."
                ),
            })

        # LLM call
        request_kwargs = dict(
            model=self.model,
            messages=messages,
            temperature=0.3,
            max_tokens=2000,
            response_format={"type": "json_object"},
        )
        # Request token logprobs for paper-aligned sequence entropy.
        request_kwargs["logprobs"] = True
        request_kwargs["top_logprobs"] = 20

        try:
            response = self.client.chat.completions.create(**request_kwargs)
        except Exception:
            # Some providers/models may not support logprobs. Fall back gracefully.
            request_kwargs.pop("logprobs", None)
            request_kwargs.pop("top_logprobs", None)
            response = self.client.chat.completions.create(**request_kwargs)

        raw = response.choices[0].message.content or "{}"
        token_count = response.usage.total_tokens if response.usage else 0
        token_top_logprobs = _extract_top_logprobs(response.choices[0])

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = {"thought": raw, "action": {"tool": "none", "input": ""}, "current_answer": raw}

        thought = parsed.get("thought", "")
        action = parsed.get("action", {})
        # Be robust to malformed action payloads, e.g. "action": "none"
        # or missing dict fields in model output.
        if isinstance(action, dict):
            tool_name = action.get("tool", "none")
            tool_input = action.get("input", "")
        elif isinstance(action, str):
            tool_name = action
            tool_input = ""
        else:
            tool_name = "none"
            tool_input = ""
        answer = parsed.get("current_answer", "")

        # Execute tool if requested
        retrieved = ""
        if tool_name and tool_name != "none" and tool_name in self.tools:
            tool_fn = self.tools[tool_name].get("function")
            if tool_fn:
                try:
                    retrieved = tool_fn(tool_input)
                except Exception as e:
                    retrieved = f"Tool error: {e}"

        return TurnState(
            turn_number=turn_number,
            answer=answer,
            retrieved_context=retrieved,
            tool_called=tool_name if tool_name != "none" else "",
            reasoning_text=thought,
            token_count=token_count,
            metadata={"token_top_logprobs": token_top_logprobs} if token_top_logprobs else {},
        )
