"""
Tool registry: provides a standard tool dictionary for agents.

Each tool is a dict with:
  - "description": str (shown to the LLM for tool selection)
  - "function": callable(str) -> str (takes input, returns observation)
"""

from __future__ import annotations

import os
import json


def make_search_tool() -> dict:
    """Web search via Tavily."""
    def search(query: str) -> str:
        from tavily import TavilyClient
        client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
        results = client.search(query=query, max_results=5)
        snippets = []
        for r in results.get("results", []):
            snippets.append(f"[{r.get('title', '')}]: {r.get('content', '')[:300]}")
        return "\n\n".join(snippets) if snippets else "No results found."

    return {
        "description": "Search the web for current information. Input: search query string.",
        "function": search,
    }


def make_wiki_tool() -> dict:
    """Wikipedia lookup."""
    def wiki_search(query: str) -> str:
        import wikipediaapi
        wiki = wikipediaapi.Wikipedia("PACE-Agent/0.1", "en")
        page = wiki.page(query)
        if page.exists():
            return page.summary[:2000]
        # Try search
        return f"No Wikipedia page found for '{query}'. Try a different search term."

    return {
        "description": "Look up a Wikipedia article. Input: article title or topic.",
        "function": wiki_search,
    }


def make_python_tool() -> dict:
    """Execute Python code in a sandbox."""
    def run_python(code: str) -> str:
        import subprocess
        try:
            result = subprocess.run(
                ["python3", "-c", code],
                capture_output=True,
                text=True,
                timeout=30,
            )
            output = result.stdout[:2000]
            if result.returncode != 0:
                output += f"\nSTDERR: {result.stderr[:500]}"
            return output or "(no output)"
        except subprocess.TimeoutExpired:
            return "Code execution timed out (30s limit)."
        except Exception as e:
            return f"Execution error: {e}"

    return {
        "description": "Execute Python code. Input: Python code string.",
        "function": run_python,
    }


def get_default_tools() -> dict[str, dict]:
    """Return the standard tool set for benchmarking."""
    return {
        "web_search": make_search_tool(),
        "wikipedia": make_wiki_tool(),
        "python": make_python_tool(),
    }


def get_frames_tools() -> dict[str, dict]:
    """Tools optimized for FRAMES (retrieval-synthesis benchmark)."""
    return {
        "web_search": make_search_tool(),
        "wikipedia": make_wiki_tool(),
    }
