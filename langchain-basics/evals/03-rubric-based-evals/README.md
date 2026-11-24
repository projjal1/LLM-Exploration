# 03 - LLM-as-a-Judge: Reference-Free Rubric Evals

Reference-free evaluations assess the quality of an LLM output against an explicit **Rubric** without requiring a ground-truth reference answer.

---

## 💡 When to Use Rubric Evals

Use reference-free rubric evals when:
- There is no single correct answer (e.g., customer service emails, conversational agents, brainstorming, summaries).
- You want to evaluate stylistic, qualitative, or constraint dimensions:
  - **Empathy & Tone**
  - **Actionability & Clarity**
  - **Conciseness**
  - **Instruction Adherence**

---

## 🎯 Best Practices for Rubrics

1. **Be Concrete**: Instead of asking the judge "Is this good?", define explicit levels:
   - *Score 1*: Defensive or ignores user complaint.
   - *Score 3*: Acknowledges issue but gives vague next steps.
   - *Score 5*: Highly empathetic, apologizes sincerely, and gives exact resolution path.

2. **Deconstruct Dimensions**: Score separate dimensions individually (e.g. Tone, Conciseness, Clarity) rather than bundling them into a single fuzzy score.

3. **Watch for Judge Biases**:
   - **Verbosity Bias**: LLMs tend to favor longer, more elaborate responses. Explicitly penalize verbosity in your prompt rubric when brevity is required.

---

## 🏃 How to Run

From this folder:
```bash
python app.py
```
