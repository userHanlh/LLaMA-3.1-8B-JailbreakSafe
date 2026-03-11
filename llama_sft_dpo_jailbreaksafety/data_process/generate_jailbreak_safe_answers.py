import json
import os
import time
import random
from openai import OpenAI

input_json_path = r""
output_json_path = r""

MAX_SAMPLES = 3079
MODEL_NAME = "gpt-5-mini"
MAX_CONSECUTIVE_ERRORS = 5
consecutive_errors = 0

SLEEP_SECONDS = 0.3   

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

MIX_RATIO_ZH = 0 # 输入英文时：输出中文（并把 query 翻译为中文）的概率


SYSTEM_PROMPT_ZH = (
    "你是一个严格遵循安全规范的大语言模型，负责生成安全回答（safe completion）。"
)

SYSTEM_PROMPT_EN = (
    "You are a safety-aligned language model that produces safe completions."
)

def build_prompt_en(query_en: str) -> str:
    return f"""
The user query below contains an inappropriate or high-risk request.

Produce a safe completion in English:
- Clearly and politely refuse
- Briefly explain why (safety, legal, or ethical reasons)
- Do NOT repeat or expand any dangerous/illegal/executable details
- Provide a safe, legal, non-operational alternative direction
- Sound natural and professional; do NOT ask the user questions
- Keep it around 100–150 English words

User query:
{query_en}

Output only the final answer (English).
""".strip()

def build_prompt_zh_from_en(query_en: str) -> str:
    schema = {
        "query_zh": "Chinese translation of the user query",
        "response_zh": "Safe completion in Chinese"
    }
    return f"""
你将收到一个英文用户请求，其中包含不恰当或高风险的内容。

你的任务：
1）将该英文请求忠实翻译为中文（保留原意与语气）
2）生成中文安全回答（safe completion），要求：
   - 明确且礼貌地拒绝
   - 简要说明拒绝原因（安全、法律或伦理）
   - 不复述、不扩展任何危险、违法或可执行的细节
   - 提供安全、合法、非操作性的替代性帮助方向
   - 语气自然、专业；不向用户提问
   - 全文控制在 150–250 字左右

只返回**合法 JSON**，schema 如下：
{json.dumps(schema, ensure_ascii=False)}

英文请求：
{query_en}
""".strip()


with open(input_json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"Loaded {len(data)} queries")

results = []
t0 = time.time()

try:
    for i, item in enumerate(data, start=0):
        query_en = item["query"].strip()

        use_chinese = (random.random() < MIX_RATIO_ZH)

        try:
            if use_chinese:
                resp = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT_ZH},
                        {"role": "user", "content": build_prompt_zh_from_en(query_en)},
                    ],
                )
                content = resp.choices[0].message.content.strip()

                try:
                    obj = json.loads(content)
                    query_out = str(obj.get("query_zh", "")).strip()
                    answer_out = str(obj.get("response_zh", "")).strip()
                except Exception:
                    query_out = query_en
                    answer_out = content

            else:
                resp = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT_EN},
                        {"role": "user", "content": build_prompt_en(query_en)},
                    ],
                )
                answer_out = resp.choices[0].message.content.strip()
                query_out = query_en

            results.append({
                "id": i,
                "type": "harmful",
                "query": query_out,
                "chosen": answer_out,
                "rejected":item["rejected"]
            })

            consecutive_errors = 0
            time.sleep(SLEEP_SECONDS)

        except Exception as e:
            consecutive_errors += 1
            print(f"[Error] index {i}:{e}")
            if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                print("Too many consecutive errors.")
                raise
            continue

except Exception as e:
    print(e)

finally:
    print("Saving results to file...")
    print("latency:", time.time() - t0)

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

