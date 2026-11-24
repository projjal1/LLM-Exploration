# 02 - LLM-as-a-Judge: Reference-Based Evals

Reference-based evaluations use an LLM as a judge to compare a model's generated answer against an authoritative **Ground Truth** reference.

---

## 💡 Why Use an LLM Judge Instead of String Matching?

Rule-based string comparisons (e.g. `answer == reference` or ROUGE/BLEU scores) suffer from severe limitations:
- **Synonyms & Paraphrasing**: "Canberra is the capital" vs "The federal capital is Canberra" would score low on exact string match.
- **Order of Information**: Models can reorder facts without losing accuracy.
- **Superficial overlap**: A hallucinated answer might share common words with the reference but say the opposite fact.

An LLM judge understands semantics and assesses whether the essential facts are accurately represented.

---

## 🧱 Key Components

1. **Structured Outputs (`with_structured_output`)**:
   Instead of parsing unstructured text or regex from the judge, we enforce a Pydantic schema:
   - `reasoning`: The judge's step-by-step thinking
   - `score`: Graded 1 to 5 Likert score
   - `unsupported_claims`: Specific hallucinations or factual errors detected
   - `is_correct`: Boolean verdict

2. **Zero-Temperature Judge**:
   The judge runs with `temperature=0.0` to minimize scoring variance and maximize consistency across benchmark runs.

---

## 🏃 How to Run

From this folder:
```bash
python app.py
```
