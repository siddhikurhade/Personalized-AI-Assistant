
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from config import MODEL_ID, MAX_NEW_TOKENS, TEMPERATURE
from prompts import SYSTEM_PROMPT

_tokenizer = None
_model = None


def load_model():
    global _tokenizer, _model

    if _model is not None:
        return _tokenizer, _model

    print(f"Loading local model: {MODEL_ID}")

    _tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        trust_remote_code=False
    )

    if _tokenizer.pad_token is None:
        _tokenizer.pad_token = _tokenizer.eos_token

    # Keep CPU execution simple and avoid GPU memory assumptions.
    _model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        trust_remote_code=False
    )

    _model.to("cpu")
    _model.eval()

    print("Model loaded successfully.")

    return _tokenizer, _model


def make_messages(question, retrieved=None, history=None):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]

    # Keep history short to reduce CPU processing and memory use.
    if history:
        for item in history[-4:]:
            if item.get("role") in {"user", "assistant"}:
                messages.append({
                    "role": item["role"],
                    "content": item["content"]
                })

    context = ""

    if retrieved:
        blocks = [
            f"[Source {i}] File: {r['source']} "
            f"(chunk {r['chunk']})\n{r['text'][:1200]}"
            for i, r in enumerate(retrieved[:2], 1)
        ]
        context = (
            "\n\nREFERENCE PASSAGES:\n"
            + "\n\n".join(blocks)
            + "\n\n"
        )

    messages.append({
        "role": "user",
        "content": context + "User question: " + question
    })

    return messages


def generate_answer(question, retrieved=None, history=None):
    tokenizer, model = load_model()

    prompt = tokenizer.apply_chat_template(
        make_messages(question, retrieved, history),
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024
    )

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True
        )

    new_tokens = output[0][inputs["input_ids"].shape[1]:]

    answer = tokenizer.decode(
        new_tokens,
        skip_special_tokens=True
    ).strip()

    if not answer:
        return "I couldn't generate an answer. Please try a shorter question."

    return answer
