import os
import torch
import platform
import warnings
import swanlab
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    pipeline,
    logging,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import BitsAndBytesConfig
from trl import SFTTrainer

# Suppress warnings for clean logs
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
logging.set_verbosity_error()

model_name = "/data_new/wyl/models/Qwen3-8B"
project_name = "qwen3-8b-finetune"
run_name = "Qwen-8b-run-2"
output_dir = f"./output/{run_name}"
lora_rank = 32
lora_alpha = 32
learning_rate = 5e-6
num_epochs = 2

os.environ["CUDA_VISIBLE_DEVICES"] = "2,3,4,5"
torch.device("cuda")

# Initialize swanlab
swanlab.init(
    project=project_name,
    config={
        "model": model_name,
        "learning_rate": learning_rate,
        "num_epochs": num_epochs,
        "hardware": platform.processor(),
        "dataset": "pubmed",
        "lora_rank": lora_rank,
        "lora_alpha": lora_alpha,
        "device": "macstudio"
    }
)
# Dataset preparation
def prepare_dataset(tokenizer):
    dataset = load_dataset("pubmed_qa", "pqa_labeled", split="train")
    
    # Split into 95% train / 5% validation
    dataset = dataset.train_test_split(train_size=0.95, test_size=0.05, seed=42)
    
    def format_abstract(sample):
        return f"Please summarize the following biomedical abstract:\n\n{sample['context']}"

    train_dataset = dataset["train"].map(
        lambda x: {"text": format_abstract(x)},
        remove_columns=dataset["train"].column_names,
        num_proc=os.cpu_count()
    )

    val_dataset = dataset["test"].map(
        lambda x: {"text": format_abstract(x)},
        remove_columns=dataset["test"].column_names,
        num_proc=os.cpu_count()
    )
    
    train_dataset = train_dataset.map(
        lambda x: tokenizer(
            x["text"],
            truncation=True,
            padding="max_length",
            max_length=4096,
            return_tensors=None,
        ),
        remove_columns=["text"],
        num_proc=os.cpu_count()
    )
    
    val_dataset = val_dataset.map(
        lambda x: tokenizer(
            x["text"],
            truncation=True,
            padding="max_length",
            max_length=4096,
            return_tensors=None,
    ),
    remove_columns=["text"],
    num_proc=os.cpu_count()
    )

    return train_dataset, val_dataset

# Model and tokenizer setup
def setup_model():
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    #load mode with 4-bit quantization
    bnb_config = BitsAndBytesConfig(
       load_in_4bit=True,
       bnb_4bit_use_double_quant=True,
       bnb_4bit_quant_type="nf4",
       bnb_4bit_compute_dtype=torch.float16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        # quantization_config=bnb_config,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        use_cache=False,
        device_map="auto",
    )

    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    return model, tokenizer

# Setup trainer
def setup_trainer(model, tokenizer, train_dataset, val_dataset):
    peft_config = LoraConfig(
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        r=lora_rank,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
    )
    model = get_peft_model(model, peft_config)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        save_strategy="epoch",
        save_total_limit=2,
        metric_for_best_model="eval_loss",
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        weight_decay=0.01,
        logging_steps=1,
        optim="adamw_torch_fused",
        report_to="swanlab",
        gradient_checkpointing=True,
        group_by_length=True,
        dataloader_num_workers=1,
        remove_unused_columns=True,
        run_name=run_name,
        #deepspeed="./deepspeed_config.json",
        per_device_train_batch_size=4,
        fp16=True,
        # resume_from_checkpoint="output/checkpoint-238",
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        peft_config=peft_config,
        args=training_args,
        data_collator=data_collator,
    )

    return trainer

# Main function

def main():
    try:
        print("\nSetting up model...")
        model, tokenizer = setup_model()

        print("\nPreparing dataset...")
        train_dataset, val_dataset = prepare_dataset(tokenizer)

        print("\nSetting up trainer...")
        trainer = setup_trainer(model, tokenizer, train_dataset, val_dataset)

        print("\nStarting training...")
        trainer.train()

        print("\nSaving LoRA adapter...")
        trainer.model.save_pretrained(f"{output_dir}/fine_tuned_model")

        print("\nMerging LoRA into base model...")
        base_model, _ = setup_model()
        merged_model = PeftModel.from_pretrained(base_model, f"{output_dir}/fine_tuned_model")
        merged_model = merged_model.merge_and_unload()

        print("\nSaving full fine-tuned model...")
        merged_model.save_pretrained(f"{output_dir}/full_finetuned_model")
        tokenizer.save_pretrained(f"{output_dir}/full_finetuned_model")

    finally:
        swanlab.finish()

if __name__ == "__main__":
    main()
