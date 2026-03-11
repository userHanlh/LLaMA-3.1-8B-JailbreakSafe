from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
import json
from datasets import Dataset, concatenate_datasets
from swanlab.integration.transformers import SwanLabCallback
from peft import PeftModel

from trl import DPOConfig, DPOTrainer

DS_CONFIG = "zero_stage2_config.json"


def load_json_to_dataset(file_path: str) -> Dataset:
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return Dataset.from_list(data)


def build_prompt_from_query(example, system_text: str):
    messages = [
        {"role": "system", "content": system_text},
        {"role": "user", "content": example["query"]},
    ]
    prompt = build_prompt_from_query.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return prompt


def map_to_dpo_fields(example, system_text: str):

    prompt = build_prompt_from_query(example, system_text=system_text)

    chosen = example["chosen"]
    rejected = example["rejected"]
    chosen = "\n" + chosen.lstrip("\n")
    rejected = "\n" + rejected.lstrip("\n")
    return {"prompt": prompt, "chosen": chosen, "rejected": rejected}


class LlamaLoraDPOPipeline:
    def __init__(self):
        self.model_name = "/nfs/huggingfacehub/Meta-Llama-3.1-8B/"
        self.tokenizer_name = "/nfs/huggingfacehub/Llama-3.1-8B-Instruct/"

        self.policy_adapter_dir = "./lora_stage2_adapter_3epoch"

        self.tokenizer = None
        self.base_model = None
        self.model = None
        self.trainer = None

        self.system_text = "You are a helpful assistant."
        self.seed = 3407

    def setup_tokenizer_and_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name, use_fast=True)

        build_prompt_from_query.tokenizer = self.tokenizer
        self.tokenizer.padding_side = "right"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.base_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
        )
        self.model = PeftModel.from_pretrained(
            self.base_model,
            self.policy_adapter_dir,
            is_trainable=True,
            adapter_name="policy",
        )
        self.model.load_adapter(self.policy_adapter_dir, adapter_name="reference")

        self.model.config.use_cache = False

    def build_dataset(
        self,
        train_paths,         
        eval_path: str | None = None,
        extra_shuffle_rounds: int = 2,
    ):
        train_datasets = []
        for p in train_paths:
            ds = load_json_to_dataset(p)
            ds = ds.map(lambda ex: map_to_dpo_fields(ex, system_text=self.system_text))
            ds = ds.remove_columns([c for c in ds.column_names if c not in ["prompt", "chosen", "rejected"]])
            train_datasets.append(ds)

        train_ds = concatenate_datasets(train_datasets)
        for i in range(extra_shuffle_rounds):
            train_ds = train_ds.shuffle(seed=self.seed + 101 * (i + 1))

        eval_ds = None
        if eval_path is not None:
            eval_ds = load_json_to_dataset(eval_path)
            eval_ds = eval_ds.map(lambda ex: map_to_dpo_fields(ex, system_text=self.system_text))
            eval_ds = eval_ds.remove_columns([c for c in eval_ds.column_names if c not in ["prompt", "chosen", "rejected"]])

        return train_ds, eval_ds

    def build_trainer(
        self,
        train_dataset,
        eval_dataset,
        output_dir,
        run_name,
        num_train_epochs=1,
        lr=1e-7,
        beta=0.1,
        max_prompt_length=1024,
        max_length=2048,
    ):
        swanlab_callback = SwanLabCallback(
            project="Llama-3.1-8B-base-lora",
            experiment_name=run_name,
            description="Stage3 DPO with TRL DPOTrainer on top of stage2 SFT LoRA (PEFT dual-adapter train/reference).only benign_data",
            config={
                "model": "Llama-3.1-8B-base",
                "tokenizer": "Llama-3.1-8B-Instruct (chat_template)",
                "policy_adapter": self.policy_adapter_dir,
                "train_data_number": len(train_dataset),
                "eval_data_number": len(eval_dataset) if eval_dataset is not None else 0,
                "beta": beta,
                "lr": lr,
                "epochs": num_train_epochs,
                "max_prompt_length": max_prompt_length,
                "max_length": max_length,
                "seed": self.seed,
            }
        )

        args = DPOConfig(
            output_dir=output_dir,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            num_train_epochs=num_train_epochs,
            learning_rate=lr,

            logging_steps=5,
            logging_first_step=True,
            save_steps=50,
            save_on_each_node=True,

            bf16=True,
            max_grad_norm=1.0,
            deepspeed=DS_CONFIG,
            seed=self.seed,

            report_to=[],
            beta=beta,
            model_adapter_name="policy",
            ref_adapter_name="reference",
            max_prompt_length=max_prompt_length,
            max_length=max_length,
            loss_type="sigmoid",
        )

        self.trainer = DPOTrainer(
            model=self.model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=self.tokenizer,
        )
        self.trainer.add_callback(swanlab_callback)

    def train_and_save(self, save_dir):
        self.trainer.train()

        if int(os.environ.get("LOCAL_RANK", "0")) == 0:
            self.trainer.model.save_pretrained(save_dir)
            self.tokenizer.save_pretrained(save_dir)
            print("saved dpo adapter:", save_dir)

    def run(
        self,
        dpo_train_paths,            
        dpo_eval_path=None,
        max_length=2048,
    ):
        train_ds, eval_ds = self.build_dataset(
            train_paths=dpo_train_paths,
            eval_path=dpo_eval_path,
            extra_shuffle_rounds=3,
        )

        self.build_trainer(
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            output_dir="./dpo_out",
            run_name="stage3_dpo_on_stage2_adapter2",
            num_train_epochs=1,
            lr=1e-6,
            beta=0.1,
            max_prompt_length=1024,
            max_length=max_length
        )

        self.train_and_save("./lora_stage3_dpo_adapter2")


if __name__ == "__main__":
    pipeline = LlamaLoraDPOPipeline()
    pipeline.setup_tokenizer_and_model()

    pipeline.run(
        dpo_train_paths=[
            "/home/hanlonghui/llama_sft/stage-3/dpo_benign_data.json",
            "/home/hanlonghui/llama_sft/stage-3/dpo_harmful_data.json",
        ],
        dpo_eval_path=None,
        max_length=2048
    )
