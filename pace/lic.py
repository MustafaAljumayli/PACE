"""
ConversationAnalyzer: processes Lost-in-Conversation JSONL logs and computes
PACE signals post-hoc.

This module bridges the LiC data format (traces with system/user/assistant/log
messages) and the PACE signal framework. It reads raw conversation logs,
reconstructs trajectories, and computes all 6 PACE signals on each turn.

Usage:
    from pace.lic import ConversationAnalyzer

    analyzer = ConversationAnalyzer()
    results = analyzer.analyze_log_file("logs/math/sharded/sharded_math_gpt-4o-mini.jsonl")
    for r in results:
        print(r["task_id"], r["trajectory"].to_dict())
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pace.embeddings import Embedder
from pace.signals import SignalComputer, HFContradictionScorer
from pace.trajectory import Trajectory, TurnState, SIGNAL_NAMES


class ConversationAnalyzer:
    """
    Analyze LiC conversation logs and compute PACE signals.

    Args:
        embedder: shared embedding model for cosine signals
        use_nli: whether to load the NLI model for contradiction detection
        signal_mask: optional subset of signals to compute (for ablation)
    """

    def __init__(
        self,
        embedder: Embedder | None = None,
        use_nli: bool = True,
        signal_mask: set[str] | None = None,
    ):
        self.embedder = embedder or Embedder()
        self.use_nli = use_nli
        self.signal_mask = signal_mask

        contradiction_scorer = None
        if use_nli:
            contradiction_scorer = HFContradictionScorer()

        self.signal_computer = SignalComputer(
            embedder=self.embedder,
            contradiction_scorer=contradiction_scorer,
            signal_mask=signal_mask,
        )

    def analyze_log_file(
        self,
        log_path: str | Path,
        show_progress: bool = True,
        output_path: str | Path | None = None,
    ) -> list[dict]:
        """
        Analyze all conversations in a LiC JSONL log file.

        Returns a list of result dicts, one per conversation, each containing:
          - task_id, task, model, is_correct, score
          - trajectory: Trajectory object with computed signals
          - signals_summary: per-signal min/max/mean
        """
        log_path = Path(log_path)
        records = self._load_records(log_path)

        if show_progress:
            try:
                from tqdm import tqdm
                records = list(tqdm(records, desc="Analyzing conversations", unit="conv"))
            except ImportError:
                records = list(records)

        results = []
        for rec in records:
            result = self.analyze_record(rec)
            if result:
                results.append(result)

        if output_path:
            self._save_results(results, Path(output_path))

        return results

    def analyze_record(self, record: dict) -> dict | None:
        """Analyze a single LiC conversation record."""
        trace = record.get("trace", [])
        if not trace:
            return None

        task_id = record.get("task_id", "")
        task = record.get("task", "")

        goal = self._extract_goal(trace)
        shards = self._extract_revealed_shards(trace)

        trajectory = Trajectory(
            goal=goal,
            episode_id=task_id,
            shards=shards,
        )

        for msg in trace:
            if msg.get("role") == "assistant":
                response_text = msg.get("content", "")
                token_top_logprobs = msg.get("token_top_logprobs", [])

                state = TurnState(
                    turn_number=len(trajectory.turns),
                    response=response_text,
                    answer=response_text,
                    metadata={"token_top_logprobs": token_top_logprobs} if token_top_logprobs else {},
                )
                trajectory.add_turn(state)
                self.signal_computer.compute(trajectory)

        return {
            "task_id": task_id,
            "task": task,
            "model": record.get("assistant_model", ""),
            "is_correct": record.get("is_correct"),
            "score": record.get("score"),
            "trajectory": trajectory,
            "signals_summary": self._summarize_signals(trajectory),
            "num_turns": len(trajectory.turns),
        }

    def _extract_goal(self, trace: list[dict]) -> str:
        """Extract the original question/goal from the first user message."""
        for msg in trace:
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, str):
                    return content
        return ""

    def _extract_revealed_shards(self, trace: list[dict]) -> list[str]:
        """Extract all shard texts that were revealed during the conversation."""
        shards = []
        for msg in trace:
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, str) and content.strip():
                    shards.append(content)
        return shards

    def _summarize_signals(self, trajectory: Trajectory) -> dict[str, dict[str, float | None]]:
        """Compute min/max/mean summary for each signal across all turns."""
        summary: dict[str, dict[str, float | None]] = {}
        for sig in SIGNAL_NAMES:
            series = trajectory.signal_series(sig)
            if series:
                summary[sig] = {
                    "min": min(series),
                    "max": max(series),
                    "mean": sum(series) / len(series),
                    "final": series[-1],
                }
            else:
                summary[sig] = {"min": None, "max": None, "mean": None, "final": None}
        return summary

    def _load_records(self, path: Path) -> list[dict]:
        """Load JSONL or pretty-printed JSON records."""
        content = path.read_text().strip()
        if not content:
            return []

        records: list[dict] = []
        if content.startswith("["):
            records = json.loads(content)
        elif content.startswith("{"):
            buf = ""
            depth = 0
            for char in content:
                buf += char
                if char == "{":
                    depth += 1
                elif char == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            records.append(json.loads(buf.strip()))
                        except json.JSONDecodeError:
                            pass
                        buf = ""
        else:
            for line in content.split("\n"):
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return records

    def _save_results(self, results: list[dict], path: Path) -> None:
        """Save analysis results as JSONL."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            for r in results:
                entry = {
                    "task_id": r["task_id"],
                    "task": r["task"],
                    "model": r["model"],
                    "is_correct": r["is_correct"],
                    "score": r["score"],
                    "num_turns": r["num_turns"],
                    "signals_summary": r["signals_summary"],
                    "trajectory": r["trajectory"].to_dict(),
                }
                f.write(json.dumps(entry) + "\n")
