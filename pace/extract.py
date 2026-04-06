"""
Robust answer extraction and evaluation for PACE experiments.

Handles the extraction bias in LiC's default pipeline where correct answers
are marked wrong due to formatting differences (commas, LaTeX, whitespace).

Strategies (tried in order):
  1. \\boxed{...}    — LaTeX boxed answers
  2. #### ...        — GSM8K gold format
  3. **...**         — Bold markdown
  4. = ...           — After equals sign
  5. Last number     — Final numeric in response

Normalization:
  - Strip commas, $, %, whitespace
  - Normalize .00 → integer
  - Strip LaTeX commands (\\text{}, \\mathrm{}, etc.)
"""

from __future__ import annotations

import re


def strip_latex(text: str) -> str:
    """Remove common LaTeX formatting from a string."""
    text = re.sub(r"\\boxed\{([^}]*)\}", r"\1", text)
    text = re.sub(r"\\text\{([^}]*)\}", r"\1", text)
    text = re.sub(r"\\mathrm\{([^}]*)\}", r"\1", text)
    text = re.sub(r"\\textbf\{([^}]*)\}", r"\1", text)
    text = re.sub(r"\\mathbf\{([^}]*)\}", r"\1", text)
    text = re.sub(r"\\frac\{([^}]*)\}\{([^}]*)\}", r"\1/\2", text)
    text = re.sub(r"\\[$%]", "", text)
    text = re.sub(r"\\\s", " ", text)
    return text.strip()


def normalize_numeric(s: str) -> str:
    """Canonicalize a numeric string for comparison."""
    s = strip_latex(s)
    s = s.replace(",", "").replace("$", "").replace("%", "").strip()
    s = re.sub(r"\s+", "", s)

    # Handle negative
    neg = s.startswith("-")
    if neg:
        s = s[1:]

    # Remove trailing .0 / .00
    s = re.sub(r"\.0+$", "", s)

    if neg and s and s != "0":
        s = "-" + s
    return s


def extract_numeric_answer(response: str) -> tuple[str, str]:
    """
    Multi-strategy extraction of a numeric answer from LLM response.
    Returns (extracted_value, method_name).
    """
    if not response:
        return "", "empty"

    # Strategy 1: \boxed{...}
    boxed = re.findall(r"\\boxed\{([^}]+)\}", response)
    if boxed:
        return normalize_numeric(boxed[-1]), "boxed"

    # Strategy 2: #### (GSM8K format)
    hashes = re.findall(r"####\s*(.+?)(?:\n|$)", response)
    if hashes:
        return normalize_numeric(hashes[-1]), "hashes"

    # Strategy 3: **bold** number
    bold = re.findall(r"\*\*([^*]*\d[^*]*)\*\*", response)
    if bold:
        # Take the last bold that contains a number
        for candidate in reversed(bold):
            nums = re.findall(r"-?[\d,]+\.?\d*", candidate)
            if nums:
                return normalize_numeric(nums[-1]), "bold"

    # Strategy 4: After equals sign (last occurrence)
    equals = re.findall(r"=\s*\$?\s*(-?[\d,]+\.?\d*)", response)
    if equals:
        return normalize_numeric(equals[-1]), "equals"

    # Strategy 5: Last number in the response
    all_nums = re.findall(r"-?[\d,]+\.?\d*", response)
    if all_nums:
        return normalize_numeric(all_nums[-1]), "last_number"

    return "", "failed"


def robust_math_eval(
    response: str,
    gold_answer: str,
    lic_extracted: str = "",
) -> dict:
    """
    Evaluate a math response against the gold answer.

    Tries LiC's extraction first (if provided), then our multi-strategy
    extraction.  Returns a standardized dict with score and method.
    """
    gold_norm = normalize_numeric(gold_answer)
    if not gold_norm:
        return {"is_correct": None, "score": None, "extraction_method": "no_gold"}

    # Check LiC extraction first (it might be right)
    if lic_extracted:
        lic_norm = normalize_numeric(lic_extracted)
        if lic_norm == gold_norm:
            return {"is_correct": True, "score": 1.0, "extraction_method": "lic_original"}

    # Our multi-strategy extraction
    extracted, method = extract_numeric_answer(response)
    if extracted == gold_norm:
        return {"is_correct": True, "score": 1.0, "extraction_method": method}

    # Check if the gold answer appears as a standalone number in the response
    # (word-boundary match to avoid "3" matching inside "300")
    import re as _re
    gold_escaped = _re.escape(gold_norm)
    if _re.search(r'(?<!\d)' + gold_escaped + r'(?!\d)', response.replace(",", "")):
        return {"is_correct": True, "score": 1.0, "extraction_method": "substring"}

    return {
        "is_correct": False,
        "score": 0.0,
        "extraction_method": method,
        "extracted": extracted,
        "gold_normalized": gold_norm,
    }
