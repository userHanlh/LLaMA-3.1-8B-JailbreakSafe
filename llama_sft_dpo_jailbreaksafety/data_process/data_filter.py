import json
import time
from pathlib import Path
from openai import OpenAI
import os

INPUT_PATH = r""
OUTPUT_PATH = r""

MODEL_NAME = "gpt-4o-mini"
MAX_SAMPLES = 5000
SLEEP_SEC = 0.3   
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

PROMPT_TEMPLATE = """You are a strict dataset quality inspector.

Task:
Given a QUESTION and its ANSWER, judge whether the ANSWER is acceptable as a training sample.

Reject the sample if:
- The answer is factually incorrect
- The answer has wrong calculations or reasoning
- The answer contradicts itself or the question
- The answer is irrelevant or misleading
- The answer is clearly incomplete
- The answer omits essential information required by the question
- The answer is excessively verbose relative to the question, without adding necessary information


Accept the sample if:
- The answer correctly addresses the question
- Minor wording issues are acceptable

IMPORTANT:
Only output ONE word:
- pass
- fail

QUESTION:
{query}

ANSWER:
{response}
"""


def judge_quality(query, response):
    prompt = PROMPT_TEMPLATE.format(query=query, response=response)

    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        result = resp.choices[0].message.content.strip().lower()
        return result == "pass"
    except Exception as e:
        print("API error:", e)
        return False  


def main():
    data = json.loads(Path(INPUT_PATH).read_text(encoding="utf-8"))

    kept = []
    checked = 0
    rejected = 0

    for item in data[23000:23000+MAX_SAMPLES]:
        checked += 1
        ok = judge_quality(item["query"], item["response"])

        if ok:
            kept.append({
                "id": len(kept),
                "type": item.get("type", "normal"),
                "query": item["query"],
                "response": item["response"],
            })
        else:
            rejected += 1

        if checked % 10 == 0:
            print(f"Checked {checked}, kept {len(kept)}, rejected {rejected}")

        time.sleep(SLEEP_SEC)

    Path(OUTPUT_PATH).write_text(
        json.dumps(kept, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    print("Done.")
    print(f"Checked: {checked}")
    print(f"Kept: {len(kept)}")
    print(f"Rejected: {rejected}")
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
