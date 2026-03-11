from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
import json
from datasets import Dataset, concatenate_datasets
from swanlab.integration.transformers import SwanLabCallback
from peft import LoraConfig, TaskType, get_peft_model
from peft import PeftModel

from trl import SFTConfig, SFTTrainer

# DeepSpeed 配置文件
DS_CONFIG = "zero_stage2_config.json"


def load_json_to_dataset(file_path: str) -> Dataset:
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return Dataset.from_list(data)


def format_to_prompt_completion(example, system_text: str):
    prompt_messages = [
        {"role": "system", "content": system_text},
        {"role": "user", "content": example["query"]},
    ]
    prompt = format_to_prompt_completion.tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    completion = example["response"] + format_to_prompt_completion.tokenizer.eos_token

    return {"prompt": prompt, "completion": completion}


class LlamaLoraStage1PackingPipeline:
    def __init__(self):
        self.model_name = "/nfs/huggingfacehub/Meta-Llama-3.1-8B/"
        self.tokenizer = None
        self.base_model = None
        self.model = None
        self.trainer = None
        self.tokenizer_name = "/nfs/huggingfacehub/Llama-3.1-8B-Instruct/"

        self.system_text = "You are a helpful assistant."
        self.seed = 3407

    def setup_model_and_lora(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name, use_fast=True)

        format_to_prompt_completion.tokenizer = self.tokenizer

        self.tokenizer.padding_side = "right"

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.base_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
        )

        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            inference_mode=False,
            r=8,
            lora_alpha=16,
            lora_dropout=0.1,
        )

        self.model = get_peft_model(self.base_model, lora_cfg)
        self.model.enable_input_require_grads()
        self.model.config.use_cache = False

    def build_dataset(self, json_paths, extra_shuffle_rounds: int = 2):
        datasets = [load_json_to_dataset(p) for p in json_paths]
        ds = concatenate_datasets(datasets)
        ds = ds.map(lambda ex: format_to_prompt_completion(ex, system_text=self.system_text))
        ds = ds.remove_columns([c for c in ds.column_names if c not in ["prompt", "completion"]])
        for i in range(extra_shuffle_rounds):
            ds = ds.shuffle(seed=self.seed + 101 * (i + 1))

        return ds

    def build_trainer(self, train_dataset, output_dir, run_name, num_train_epochs=1, lr=2e-4, max_seq_length=3072):
        swanlab_callback = SwanLabCallback(
            project="Llama-3.1-8B-base-lora",
            experiment_name=run_name,
            description="Stage2 LoRA SFT with TRL SFTTrainer on prompt+completion dataset (default completion-only loss).",
            config={
                "model": "Llama-3.1-8B-base",
                "train_data_number": len(train_dataset),
                "lora_rank": 8,
                "lora_alpha": 16,
                "lora_dropout": 0.1,
                "lr": lr,
                "epochs": num_train_epochs,
                "packing": False,
                "max_seq_length": max_seq_length,
                "seed": self.seed,
                "extra_shuffle_rounds": 3,
                "loss_mask": "default_completion_only_loss",
                "tokenizer": "Llama-3.1-8B-Instruct (chat_template)",
            }
        )

        args = SFTConfig(
            output_dir=output_dir,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            num_train_epochs=num_train_epochs,
            learning_rate=lr,
            lr_scheduler_type="cosine",
            warmup_ratio=0.04,
            logging_steps=10,
            logging_first_step=True,
            save_steps=200,
            save_on_each_node=True,
            gradient_checkpointing=True,
            report_to=[],
            bf16=True,
            max_grad_norm=1.0,
            deepspeed=DS_CONFIG,
            seed=self.seed,
            packing=False,
            max_length=max_seq_length,
            completion_only_loss=True,
        )

        self.trainer = SFTTrainer(
            model=self.model,
            args=args,
            train_dataset=train_dataset,
            processing_class=self.tokenizer,
        )

        self.trainer.add_callback(swanlab_callback)

    def train_and_save(self, save_dir):
        self.trainer.train()
        self.trainer.save_model(save_dir)
        self.trainer.save_state()

    def run_stage1(self,
                   boundary_path="stage-1/boundary_benign.json",
                   explicit_path="stage-1/explicit_harmful.json",
                   max_length=3072):
        train_ds = self.build_dataset(
            [boundary_path, explicit_path],
            extra_shuffle_rounds=3,  
        )


        self.build_trainer(
            train_dataset=train_ds,
            output_dir="./lora_stage1_out",
            run_name="stage1_boundary+explicit_prompt_completion_shuffle",
            num_train_epochs=1,
            lr=2e-4,
            max_seq_length=max_length
        )

        self.train_and_save("./lora_stage1_adapter")

    def load_stage1_adapter_for_stage2(self, stage1_adapter_dir="./lora_stage1_adapter"):
        self.model = PeftModel.from_pretrained(
            self.base_model,
            stage1_adapter_dir,
            is_trainable=True,
        )
        self.model.enable_input_require_grads()
        self.model.config.use_cache = False
    
    def run_stage2(self,
                   boundary_path="stage-2/boundary_benign.json",
                   explicit_path="stage-2/explicit_harmful.json",
                   jailbreak_path="stage-2/jailbreak_harmful.json",
                   pseudo_path="stage-2/pseudo_jailbreak_Benign.json",
                   stage1_adapter_dir="./lora_stage1_adapter",
                   max_length=2048):
        self.load_stage1_adapter_for_stage2(stage1_adapter_dir=stage1_adapter_dir)

        train_ds = self.build_dataset(
            [jailbreak_path, pseudo_path, explicit_path, boundary_path],
            extra_shuffle_rounds=3,
        )
        self.build_trainer(
            train_dataset=train_ds,
            output_dir="./lora_stage2_out_4epoch",
            run_name="stage2_jailbreak+pseudo+explicit+boundary_prompt_completion_shuffle",
            num_train_epochs=4,
            lr=1e-4,
            max_seq_length=max_length
        )

        self.train_and_save("./lora_stage2_adapter_4epoch")


if __name__ == "__main__":
    pipeline = LlamaLoraStage1PackingPipeline()
    pipeline.setup_model_and_lora()

    # pipeline.run_stage1(
    #     boundary_path="stage-1/boundary_benign.json",
    #     explicit_path="stage-1/explicit_harmful.json",
    #     max_length=2048
    # )
    pipeline.run_stage2(
        boundary_path="stage-2/boundary_benign.json",
        explicit_path="stage-2/explicit_harmful.json",
        jailbreak_path="stage-2/jailbreak_harmful.json",
        pseudo_path="stage-2/pseudo_jailbreak_Benign.json",
        stage1_adapter_dir="./lora_stage1_adapter",
        max_length=2048
    )


