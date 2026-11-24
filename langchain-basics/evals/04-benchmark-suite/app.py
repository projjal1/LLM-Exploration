"""
04: End-to-End Evaluation Benchmark Suite & Scorecard
====================================================

Concept:
--------
In production LLM applications, evals are not run one-by-one manually.
Instead, an automated test harness runs a benchmark dataset of test cases,
computes aggregate metrics, and alerts engineers to regressions.

This script demonstrates:
1. Defining a curated evaluation dataset of test cases.
2. Routing each test case to its designated evaluator (Rule-based, Reference Judge, or Rubric Judge).
3. Measuring latency and pass/fail metrics.
4. Printing an executive scorecard and saving results to `eval_report.json`.
"""

import json
import os
import time
from typing import Any, Dict, List
from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate


# ---------------------------------------------------------------------------
# 1. Models & Evaluator Schemas
# ---------------------------------------------------------------------------

generator_llm = ChatOllama(model="qwen3:8b", temperature=0.2)
judge_llm = ChatOllama(model="qwen3:8b", temperature=0.0)


class FactualJudgeOutput(BaseModel):
    reasoning: str = Field(description="Step-by-step reasoning comparing candidate against reference")
    score: int = Field(ge=1, le=5, description="Factual accuracy score 1-5")
    passed: bool = Field(description="True if facts are accurate and consistent with the reference")


class RubricJudgeOutput(BaseModel):
    reasoning: str = Field(description="Evaluation of candidate against the rubric criteria")
    clarity_score: int = Field(ge=1, le=5, description="Clarity rating 1-5")
    adherence_score: int = Field(ge=1, le=5, description="Instruction adherence rating 1-5")
    passed: bool = Field(description="True if both scores >= 3")


# Judge Chains
factual_chain = (
    ChatPromptTemplate.from_messages([
        ("system", "You are an impartial judge. Compare candidate answer to reference truth."),
        ("user", "Question: {q}\nReference: {ref}\nCandidate: {cand}\nEvaluate factual correctness.")
    ])
    | judge_llm.with_structured_output(FactualJudgeOutput)
)

rubric_chain = (
    ChatPromptTemplate.from_messages([
        ("system", "You are an expert judge. Evaluate the candidate according to the rubric."),
        ("user", "Task: {task}\nRubric: {rubric}\nCandidate: {cand}\nGrade the candidate response.")
    ])
    | judge_llm.with_structured_output(RubricJudgeOutput)
)


# ---------------------------------------------------------------------------
# 2. Evaluation Dataset
# ---------------------------------------------------------------------------

BENCHMARK_DATASET: List[Dict[str, Any]] = [
    {
        "id": "TC-FACT-01",
        "category": "Factual QA",
        "eval_type": "reference",
        "prompt": "What is the boiling point of water in Celsius and Fahrenheit at standard atmospheric pressure?",
        "reference": "At standard sea-level pressure, water boils at 100 degrees Celsius and 212 degrees Fahrenheit.",
    },
    {
        "id": "TC-FACT-02",
        "category": "Factual QA",
        "eval_type": "reference",
        "prompt": "Which planet is known as the Red Planet and why?",
        "reference": "Mars is known as the Red Planet due to large amounts of iron oxide (rust) on its surface.",
    },
    {
        "id": "TC-RULE-01",
        "category": "Rule-Based Format",
        "eval_type": "rule_keywords",
        "prompt": "State Newton's First Law of Motion in one sentence. You must include the words 'inertia' and 'motion'.",
        "required_keywords": ["inertia", "motion"],
        "max_words": 30,
    },
    {
        "id": "TC-RUBRIC-01",
        "category": "Qualitative Rubric",
        "eval_type": "rubric",
        "prompt": "Explain quantum superposition to a 10-year-old using a coin analogy.",
        "rubric": "Uses spinning coin analogy, age-appropriate language, clear and engaging explanation.",
    },
]


# ---------------------------------------------------------------------------
# 3. Benchmark Runner
# ---------------------------------------------------------------------------

def run_suite():
    print("=" * 80)
    print("🚀 EXECUTING LLM BENCHMARK EVALUATION SUITE")
    print(f"Target Model: qwen3:8b | Judge: qwen3:8b | Total Cases: {len(BENCHMARK_DATASET)}")
    print("=" * 80)

    results = []

    for idx, test in enumerate(BENCHMARK_DATASET, start=1):
        print(f"\n[{idx}/{len(BENCHMARK_DATASET)}] Case {test['id']}: {test['category']}")
        print(f"Prompt: {test['prompt']}")

        # Time model generation
        start_t = time.time()
        response_text = generator_llm.invoke(test["prompt"]).content.strip()
        latency = time.time() - start_t

        print(f"Response ({latency:.2f}s):\n{response_text[:140]}..." if len(response_text) > 140 else f"Response:\n{response_text}")

        # Dispatch evaluator
        eval_t0 = time.time()
        passed = False
        meta = {}

        if test["eval_type"] == "reference":
            judge_res: FactualJudgeOutput = factual_chain.invoke({
                "q": test["prompt"],
                "ref": test["reference"],
                "cand": response_text,
            })
            passed = judge_res.passed
            meta = {
                "score": f"{judge_res.score}/5",
                "reasoning": judge_res.reasoning,
            }

        elif test["eval_type"] == "rubric":
            rubric_res: RubricJudgeOutput = rubric_chain.invoke({
                "task": test["prompt"],
                "rubric": test["rubric"],
                "cand": response_text,
            })
            passed = rubric_res.passed
            meta = {
                "clarity": f"{rubric_res.clarity_score}/5",
                "adherence": f"{rubric_res.adherence_score}/5",
                "reasoning": rubric_res.reasoning,
            }

        elif test["eval_type"] == "rule_keywords":
            text_lower = response_text.lower()
            missing = [k for k in test["required_keywords"] if k.lower() not in text_lower]
            word_count = len(response_text.split())
            length_ok = word_count <= test.get("max_words", 999)
            passed = (len(missing) == 0) and length_ok
            meta = {
                "missing_keywords": missing,
                "word_count": word_count,
                "max_words": test.get("max_words"),
            }

        eval_latency = time.time() - eval_t0
        status_label = "✅ PASS" if passed else "❌ FAIL"
        print(f"Verdict: {status_label} (Eval took {eval_latency:.2f}s)")

        results.append({
            "id": test["id"],
            "category": test["category"],
            "eval_type": test["eval_type"],
            "passed": passed,
            "gen_latency": round(latency, 2),
            "eval_latency": round(eval_latency, 2),
            "meta": meta,
        })

    # -----------------------------------------------------------------------
    # 4. Final Scorecard Reporting
    # -----------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("📊 BENCHMARK SCORECARD SUMMARY")
    print("=" * 80)
    print(f"{'ID':<15} {'Category':<22} {'Verdict':<10} {'Latency':<10} {'Notes'}")
    print("-" * 80)

    for r in results:
        v_str = "✅ PASS" if r["passed"] else "❌ FAIL"
        notes = str(r["meta"].get("score") or r["meta"].get("clarity") or "Rule Check")
        print(f"{r['id']:<15} {r['category']:<22} {v_str:<10} {r['gen_latency']}s{'':<5} {notes}")

    passed_count = sum(1 for r in results if r["passed"])
    total_count = len(results)
    pass_rate = (passed_count / total_count) * 100
    avg_latency = sum(r["gen_latency"] for r in results) / total_count

    print("-" * 80)
    print(f"Overall Pass Rate: {passed_count}/{total_count} ({pass_rate:.1f}%)")
    print(f"Average Generation Latency: {avg_latency:.2f}s")
    print("=" * 80)

    # Save to JSON
    report_file = os.path.join(os.path.dirname(__file__), "eval_report.json")
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model": "qwen3:8b",
                "pass_rate": pass_rate,
                "passed_count": passed_count,
                "total_count": total_count,
                "avg_latency": avg_latency,
                "results": results,
            },
            f,
            indent=2,
        )
    print(f"Full report exported to: {report_file}")


if __name__ == "__main__":
    run_suite()
