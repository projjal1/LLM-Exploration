# 04 - End-to-End Evaluation Benchmark Suite

This module combines rule-based and LLM-as-a-judge evaluators into a unified benchmark harness with automated scoring, latency measurement, and scorecard generation.

---

## 💡 Why Build a Benchmark Suite?

When updating prompts, fine-tuning, changing system instructions, or switching model versions, you need to ensure changes don't break existing capabilities.

A benchmark suite:
- Runs an evaluation dataset representing critical tasks.
- Routes each test case to its designated evaluator.
- Calculates an overall **Pass Rate** and **Average Latency**.
- Emits structured results to `eval_report.json` for CI/CD tracking.

---

## 📊 Scorecard Metrics

- **Pass Rate**: Percentage of test cases meeting all criteria.
- **Latency Tracking**: Generation vs evaluation latency per item.
- **Structured Failure Analysis**: Detailed judge reasoning preserved in JSON export.

---

## 🏃 How to Run

From this folder:
```bash
python app.py
```
This generates `eval_report.json` in the same directory.
