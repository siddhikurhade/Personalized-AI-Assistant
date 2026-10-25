"""Optional LoRA/QLoRA fine-tuning.

Install in a suitable environment:
pip install peft trl datasets bitsandbytes

Each JSONL line: {"instruction":"...","input":"...","output":"..."}
"""
import argparse
import json
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", default=os.getenv("MODEL_ID", "Qwen/Qwen2.5-3B-Instruct"))
    parser.add_argument("--train_file", default="data/train_sample.jsonl")
    parser.add_argument("--output_dir", default="outputs/qwen-lora")
    parser.add_argument("--max_seq_length", type=int, default=768)
    parser.add_argument("--epochs", type=float, default=2)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=8)
    args = parser.parse_args()

    try:
        import torch
        from datasets import Dataset
        from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
        from peft import LoraConfig, prepare_model_for_kbit_training
        from trl import SFTTrainer
    except ImportError as exc:
        raise SystemExit("Install optional dependencies: pip install peft trl datasets bitsandbytes") from exc

    rows = []
    with open(args.train_file, encoding="utf-8") as f:
        rows.extend(json.loads(line) for line in f if line.strip())
    if not rows:
        raise SystemExit("Training file has no examples.")

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    qconfig = None
    use_qlora = False
    if torch.cuda.is_available():
        try:
            import bitsandbytes  # noqa: F401
            from transformers import BitsAndBytesConfig
            qconfig = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
            use_qlora = True
        except Exception:
            pass

    model_kwargs = {"device_map": "auto"} if torch.cuda.is_available() else {}
    if qconfig:
        model_kwargs["quantization_config"] = qconfig
    else:
        model_kwargs["torch_dtype"] = torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.model_id, **model_kwargs)
    if use_qlora:
        model = prepare_model_for_kbit_training(model)

    def format_row(row):
        user = row.get("instruction", "")
        if row.get("input"):
            user += "\n\nContext/input:\n" + row["input"]
        messages = [{"role": "system", "content": "You are a helpful, accurate personalized AI assistant."},
                    {"role": "user", "content": user},
                    {"role": "assistant", "content": row.get("output", "")}]
        return {"text": tokenizer.apply_chat_template(messages, tokenize=False)}

    dataset = Dataset.from_list([format_row(row) for row in rows])
    lora = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05, bias="none",
                      task_type="CAUSAL_LM", target_modules="all-linear")
    train_args = TrainingArguments(
        output_dir=args.output_dir, num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum, learning_rate=2e-4,
        logging_steps=5, save_strategy="epoch", report_to="none",
        fp16=torch.cuda.is_available(), gradient_checkpointing=True,
        optim="paged_adamw_8bit" if use_qlora else "adamw_torch",
        remove_unused_columns=False,
    )
    trainer = SFTTrainer(model=model, args=train_args, train_dataset=dataset,
                         peft_config=lora, processing_class=tokenizer,
                         max_seq_length=args.max_seq_length)
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved adapter/model to {args.output_dir}")
    print("Evaluate against the base model on held-out questions before claiming improvement.")

if __name__ == "__main__":
    main()
