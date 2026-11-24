"""
01: Deterministic & Rule-Based Evaluations
==========================================

Concept:
--------
Rule-based evaluations test an LLM's response using deterministic logic (code, regex,
schemas) without calling an LLM judge.

Advantages:
- Extremely fast (microseconds)
- Zero extra LLM API / token cost
- 100% reproducible and deterministic
- Essential for syntax, schemas, safety blocklists, and hard business constraints

Techniques demonstrated:
1. Keyword Assertion (presence of required terms, absence of forbidden terms)
2. Word Count / Length Range Assertion
3. Regex Pattern Matching (e.g., format compliance)
4. Strict Pydantic JSON Schema Adherence
"""

import json
import re
from typing import List, Optional, Type
from pydantic import BaseModel, Field, ValidationError
from langchain_ollama import ChatOllama


# ---------------------------------------------------------------------------
# 1. Evaluator Functions
# ---------------------------------------------------------------------------

def eval_keywords(
    text: str,
    required: Optional[List[str]] = None,
    forbidden: Optional[List[str]] = None,
) -> dict:
    """Checks that all required keywords exist and no forbidden keywords are present."""
    text_lower = text.lower()
    # Check for missing required keywords (case-insensitive)
    missing_required = [kw for kw in (required or []) if kw.lower() not in text_lower]
    # Check for forbidden keywords (case-insensitive)
    present_forbidden = [kw for kw in (forbidden or []) if kw.lower() in text_lower]

    # Determine overall pass/fail status
    passed = len(missing_required) == 0 and len(present_forbidden) == 0
    return {
        "eval_type": "keywords",
        "passed": passed,
        "missing_required": missing_required,
        "present_forbidden": present_forbidden,
    }


def eval_word_length(text: str, min_words: int = 1, max_words: int = 100) -> dict:
    """Verifies that response length falls within specified word boundaries."""
    words = text.split()
    word_count = len(words)
    passed = min_words <= word_count <= max_words
    return {
        "eval_type": "word_length",
        "passed": passed,
        "word_count": word_count,
        "allowed_range": (min_words, max_words),
    }


def eval_regex(text: str, pattern: str, pattern_description: str) -> dict:
    """Tests if the response matches a specific regular expression pattern."""
    match = re.search(pattern, text)
    return {
        "eval_type": "regex_match",
        "pattern_description": pattern_description,
        "passed": bool(match),
        "matched_content": match.group(0) if match else None,
    }


def eval_pydantic_schema(raw_text: str, schema_class: Type[BaseModel]) -> dict:
    """
    Strips markdown code fences and validates JSON against a Pydantic model.
    Catches syntax errors, missing fields, and type mismatches.
    """
    clean_text = raw_text.strip()
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", clean_text)
    if match:
        clean_text = match.group(1).strip()

    try:
        data = json.loads(clean_text)
        validated = schema_class.model_validate(data)
        return {
            "eval_type": "json_schema",
            "passed": True,
            "parsed": validated.model_dump(),
            "error": None,
        }
    except (json.JSONDecodeError, ValidationError) as err:
        return {
            "eval_type": "json_schema",
            "passed": False,
            "parsed": None,
            "error": str(err),
        }


# ---------------------------------------------------------------------------
# 2. Demonstration with qwen3:8b
# ---------------------------------------------------------------------------

class BookRecommendation(BaseModel):
    title: str = Field(description="Book title")
    author: str = Field(description="Author name")
    publication_year: int = Field(description="Year published")
    genres: List[str] = Field(description="List of genres")


def main():
    print("=" * 70)
    print("🧪 RUNNING DETERMINISTIC / RULE-BASED EVALS")
    print("=" * 70)

    llm = ChatOllama(model="qwen3:8b", temperature=0.1)

    # --- Test 1: Keyword & Length Constraints ---
    print("\n[Test 1] Keyword and Length Constraints on Summarization")
    prompt_1 = "Provide a 1-sentence definition of Machine Learning. Do not use the word 'magic'."
    response_1 = llm.invoke(prompt_1).content.strip()
    print(f"Prompt: {prompt_1}")
    print(f"Response: {response_1}\n")

    kw_result = eval_keywords(
        response_1,
        required=["learn", "data"],
        forbidden=["magic", "sentient"],
    )
    len_result = eval_word_length(response_1, min_words=5, max_words=35)

    print(f"Keyword Eval:   {'✅ PASSED' if kw_result['passed'] else '❌ FAILED'}")
    if not kw_result['passed']:
        print(f"   Missing: {kw_result['missing_required']}, Forbidden found: {kw_result['present_forbidden']}")

    print(f"Length Eval:    {'✅ PASSED' if len_result['passed'] else '❌ FAILED'} ({len_result['word_count']} words, allowed {len_result['allowed_range']})")

    # --- Test 2: Regex Format Verification ---
    print("\n" + "-" * 70)
    print("[Test 2] Regex Pattern Matching (ISO Date extraction)")
    prompt_2 = "What date was the Apollo 11 moon landing? Provide the date strictly in YYYY-MM-DD format."
    response_2 = llm.invoke(prompt_2).content.strip()
    print(f"Prompt: {prompt_2}")
    print(f"Response: {response_2}\n")

    regex_result = eval_regex(response_2, r"\b\d{4}-\d{2}-\d{2}\b", "ISO Date (YYYY-MM-DD)")
    print(f"Regex Eval:     {'✅ PASSED' if regex_result['passed'] else '❌ FAILED'}")
    print(f"   Matched: {regex_result['matched_content']}")

    # --- Test 3: Structured Pydantic Schema Validation ---
    print("\n" + "-" * 70)
    print("[Test 3] Pydantic Schema Validation")
    prompt_3 = (
        "Recommend one famous science fiction book in valid JSON format. "
        "Keys must be: 'title' (string), 'author' (string), 'publication_year' (integer), 'genres' (list of strings). "
        "Return ONLY the JSON object."
    )
    response_3 = llm.invoke(prompt_3).content.strip()
    print(f"Prompt: {prompt_3}")
    print(f"Response: {response_3}\n")

    schema_result = eval_pydantic_schema(response_3, BookRecommendation)
    print(f"Schema Eval:    {'✅ PASSED' if schema_result['passed'] else '❌ FAILED'}")
    if schema_result['passed']:
        print(f"   Parsed Payload: {schema_result['parsed']}")
    else:
        print(f"   Error Details: {schema_result['error']}")

    print("\n" + "=" * 70)
    print("Done! Rule-based evals allow instant, cost-free validation.")
    print("=" * 70)


if __name__ == "__main__":
    main()
