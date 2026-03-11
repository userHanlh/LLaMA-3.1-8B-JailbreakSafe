import json, time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL = "/nfs/huggingfacehub/Meta-Llama-3.1-8B/"
TOKENIZER = "/nfs/huggingfacehub/Llama-3.1-8B-Instruct/"
ADAPTER_DIR = "./lora_stage2_adapter_3epoch"

VAL_JSON = "stage_dpo/benign_query2.json"
OUT_JSON = "stage_dpo/benign_data2.json"

SYSTEM_TEXT = "You are a helpful assistant."

BATCH_SIZE = 8   #batch 大小

def load_val(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

import torch
import torch.nn.functional as F

@torch.inference_mode()
def answer_batch(model, tokenizer, queries, max_new_tokens=256):
    messages = [
        [
            {"role": "system", "content": SYSTEM_TEXT},
            {"role": "user", "content": q},
        ]
        for q in queries
    ]

    enc = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        padding=True,
        return_dict=True,   
    )
    enc = {k: v.to(model.device) for k, v in enc.items()}

    gen_out = model.generate(
        input_ids=enc["input_ids"],
        attention_mask=enc["attention_mask"],  
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True,
        num_beams=1,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        repetition_penalty=1.1,
        no_repeat_ngram_size=4,
    )

    max_len = enc["input_ids"].shape[1] 

    results = []
    for i in range(len(queries)):
        gen = gen_out[i][max_len:]  
        text = tokenizer.decode(gen, skip_special_tokens=True).strip()

        low = text.lower()
        for m in [
            "\n\nuser", "\n\nassistant",
            "\nuser", "\nassistant",
            "��user", "��assistant",
            "�user", "�assistant", "�"
        ]:
            j = low.find(m)
            if j != -1:
                text = text[:j].strip()
                break

        results.append(text)

    return results



def main():
    tok = AutoTokenizer.from_pretrained(TOKENIZER, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base, ADAPTER_DIR)
    model.eval()

    print("eos_token:", tok.eos_token, tok.eos_token_id)
    print("special tokens:", tok.special_tokens_map)
    print("eos:", tok.eos_token, tok.eos_token_id)
    print("eot_id:", tok.convert_tokens_to_ids("<|eot_id|>"))

    data = load_val(VAL_JSON)
    out = []

    for i in range(0, len(data), BATCH_SIZE):
        batch = data[i:i + BATCH_SIZE]
        queries = [ex["query"] for ex in batch]

        answers = answer_batch(model, tok, queries)

        for ex, a in zip(batch, answers):
            out.append({
                "id": ex.get("id", str(i)),
                "type": ex["type"],
                "query": ex["query"],
                "raw_answer":ex["response"],
                "gen_answer": a,
                "meta": {
                    "model": "Meta-Llama-3.1-8B + LoRA(stage2)",
                    "ts": time.time()
                }
            })

        if (i + BATCH_SIZE) % 20 == 0:
            print(f"done {min(i + BATCH_SIZE, len(data))}/{len(data)}")

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print("saved:", OUT_JSON)


if __name__ == "__main__":
    main()

