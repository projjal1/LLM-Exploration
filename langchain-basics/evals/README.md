# LangChain Evals with Ollama (`qwen3:8b`)

This directory breaks down LLM evaluation into 4 focused modules, each with its own runnable script and guide:

---

## 📁 Modules

1. [01-rule-based-evals](file:///f:/Learning/Code-Artifacts/ai-codes/langchain-basics/evals/01-rule-based-evals/README.md)
   - **What it tests**: Hard rules, constraints, regex patterns, and strict Pydantic JSON schemas.
   - **Characteristics**: Microsecond speed, zero judge token cost, 100% reproducible.

2. [02-reference-based-evals](file:///f:/Learning/Code-Artifacts/ai-codes/langchain-basics/evals/02-reference-based-evals/README.md)
   - **What it tests**: Factual correctness, semantic similarity, and hallucination detection against a known ground truth.
   - **Characteristics**: Uses an LLM judge (`qwen3:8b`) with structured Pydantic scoring (1–5 Likert scale + reasoning).

3. [03-rubric-based-evals](file:///f:/Learning/Code-Artifacts/ai-codes/langchain-basics/evals/03-rubric-based-evals/README.md)
   - **What it tests**: Open-ended qualitative criteria (clarity, conciseness, tone, empathy) where no single reference answer exists.
   - **Characteristics**: Multi-criteria rubric grading with structured critique.

4. [04-benchmark-suite](file:///f:/Learning/Code-Artifacts/ai-codes/langchain-basics/evals/04-benchmark-suite/README.md)
   - **What it tests**: End-to-end test suite runner that executes multiple test cases, calculates pass rates and latencies, prints a scorecard, and exports `eval_report.json`.

---

## 🚀 Setup

```bash
pip install -r requirements.txt
```
Make sure Ollama is running with `qwen3:8b`:
```bash
ollama run qwen3:8b
```
