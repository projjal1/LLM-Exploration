# 01 - Deterministic & Rule-Based Evals

Deterministic evaluations evaluate model responses using deterministic code and schemas without involving another LLM.

---

## 💡 Key Concepts

1. **Exact & Substring Matching**:
   - Verify that necessary domain terminology exists (e.g. required keywords).
   - Ensure negative constraints / forbidden terms (e.g., profanity, competitor names) are excluded.

2. **Length & Budget Constraints**:
   - Check minimum / maximum word counts or character budgets.

3. **Pattern Verification (Regex)**:
   - Ensure specific formats like dates (`YYYY-MM-DD`), email addresses, currency codes, or IDs match exact regex patterns.

4. **Schema Adherence (Pydantic)**:
   - Validate structured responses against a Pydantic model (`model_validate`).
   - Catches syntax errors, missing fields, or incorrect types.

---

## ⚖️ Trade-offs

| Pros | Cons |
|---|---|
| ⚡ Microsecond execution speed | ❌ Cannot evaluate nuance, tone, or semantic meaning |
| 💰 Zero additional token cost | ❌ Brittle to paraphrasing or synonyms |
| 🎯 100% deterministic & reproducible | |

---

## 🏃 How to Run

From this folder:
```bash
python app.py
```
