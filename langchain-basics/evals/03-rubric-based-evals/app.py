"""
03: LLM-as-a-Judge (Reference-Free Rubric Evaluation)
===================================================

Concept:
--------
In many practical LLM applications (customer support, summarization, creative drafting,
chatbots), there is NO single "correct" reference answer.

Instead of comparing against ground truth, we evaluate against a **Rubric**:
a structured set of scoring criteria such as:
1. Instruction Adherence (Did it follow all constraints?)
2. Clarity & Readability (Is it coherent and easy to understand?)
3. Tone & Politeness (Is it professional, empathetic, or appropriately styled?)
4. Conciseness (Is it direct without fluff?)

Key Techniques:
- Multi-criteria Pydantic scoring schema.
- Reference-free qualitative judging.
"""

from typing import List
from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate


# ---------------------------------------------------------------------------
# 1. Models & Schemas
# ---------------------------------------------------------------------------

generator_llm = ChatOllama(model="qwen3:8b", temperature=0.4)
judge_llm = ChatOllama(model="qwen3:8b", temperature=0.0)


class CriterionScore(BaseModel):
    """Score and critique for an individual rubric dimension."""
    criterion_name: str = Field(description="Name of the evaluated criterion (e.g. Clarity, Empathy, Conciseness)")
    score: int = Field(ge=1, le=5, description="Rating from 1 (poor) to 5 (excellent)")
    critique: str = Field(description="Brief explanation of why this score was assigned")


class RubricEvaluationReport(BaseModel):
    """Full rubric evaluation report returned by the judge."""
    summary_critique: str = Field(description="Overall evaluation summary of the response")
    scores: List[CriterionScore] = Field(description="Scores for each criterion evaluated")
    average_score: float = Field(description="Calculated or estimated average score across criteria")
    passed: bool = Field(description="True if the response meets or exceeds acceptable standards (e.g. avg >= 3.5)")


# ---------------------------------------------------------------------------
# 2. Judge Prompt & Chain
# ---------------------------------------------------------------------------

RUBRIC_JUDGE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an expert, objective evaluation judge evaluating conversational and task outputs.\n"
        "Evaluate the candidate response strictly according to the user instructions and the provided rubric.\n"
        "Grade each criterion independently on a 1-5 scale:\n"
        "1 = Poor / Failed entirely\n"
        "2 = Subpar / Major issues\n"
        "3 = Acceptable / Meets bare minimum\n"
        "4 = Good / Minor flaws\n"
        "5 = Outstanding / Exemplary\n"
        "Provide honest, critical scores and actionable critique."
    ),
    (
        "user",
        "User Prompt / Task:\n{prompt}\n\n"
        "Candidate Response to Evaluate:\n{candidate_response}\n\n"
        "Rubric Criteria to Grade:\n{rubric_criteria}\n\n"
        "Generate the evaluation report."
    )
])

rubric_judge_chain = RUBRIC_JUDGE_PROMPT | judge_llm.with_structured_output(RubricEvaluationReport)


def evaluate_with_rubric(prompt: str, candidate_response: str, rubric_criteria: str) -> RubricEvaluationReport:
    """Invokes the rubric evaluator chain."""
    return rubric_judge_chain.invoke({
        "prompt": prompt,
        "candidate_response": candidate_response,
        "rubric_criteria": rubric_criteria,
    })


# ---------------------------------------------------------------------------
# 3. Demonstration Runs
# ---------------------------------------------------------------------------

def main():
    print("=" * 75)
    print("📋 RUNNING REFERENCE-FREE RUBRIC EVALUATION")
    print("=" * 75)

    test_scenarios = [
        {
            "name": "Customer Support Escalation Response",
            "prompt": (
                "A customer writes: 'My package arrived 4 days late and the box was torn open! "
                "I want an immediate refund or I am canceling my subscription!'\n"
                "Respond as a customer service representative. Be apologetic, empathetic, "
                "and explain the next steps clearly."
            ),
            "rubric": (
                "- Empathy & Tone: Apologetic and de-escalating without being defensive.\n"
                "- Clarity of Resolution: Clearly states actions being taken or next steps.\n"
                "- Professionalism: Courteous and representative of a customer-first brand."
            ),
        },
        {
            "name": "Executive Summary for Leadership",
            "prompt": (
                "Summarize the business impact of migrating from on-premises servers to cloud infrastructure "
                "for a non-technical Chief Financial Officer (CFO). Keep it under 4 bullet points."
            ),
            "rubric": (
                "- Conciseness: Strictly under 4 bullet points, no fluff.\n"
                "- Financial / Business Relevance: Highlights CapEx vs OpEx, cost efficiency, or agility.\n"
                "- Jargon-free Clarity: Suitable for an executive who does not code."
            ),
        }
    ]

    for idx, test in enumerate(test_scenarios, start=1):
        print(f"\n[Scenario {idx}] {test['name']}")
        print(f"Task: {test['prompt']}\n")

        print("Generating response with qwen3:8b...")
        response = generator_llm.invoke(test["prompt"]).content.strip()
        print(f"Candidate Response:\n{response}\n")

        print("Running Rubric Evaluation with Judge...")
        report: RubricEvaluationReport = evaluate_with_rubric(
            prompt=test["prompt"],
            candidate_response=response,
            rubric_criteria=test["rubric"],
        )

        status = "✅ PASSED" if report.passed else "❌ FAILED"
        print(f"\nVerdict:          {status} (Overall Avg: {report.average_score:.1f}/5)")
        print(f"Summary Critique: {report.summary_critique}")
        print("\nCriteria Breakdown:")
        for crit in report.scores:
            print(f"  • {crit.criterion_name:<25}: {crit.score}/5 -> {crit.critique}")
        print("-" * 75)


if __name__ == "__main__":
    main()
