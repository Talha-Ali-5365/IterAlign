import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import snapshot_download


def ensure_model_downloaded(model_repo: str, local_dir: str) -> str:
    local_path = Path(local_dir)
    if not local_path.exists() or not any(local_path.iterdir()):
        print(f"Downloading model {model_repo} to {local_dir}...")
        snapshot_download(repo_id=model_repo, local_dir=local_dir)
        print("Model downloaded successfully.")
    else:
        print(f"Model found at {local_dir}.")
    return local_dir


def load_model_and_tokenizer(model_path: str, dtype=torch.float16, device_map="auto"):
    print(f"Loading model from {model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map=device_map,
    )
    model.eval()
    print("Model loaded.")
    return model, tokenizer


def generate_responses(model, tokenizer, prompts: list[str], max_new_tokens: int = 300) -> list[str]:
    tokenizer.padding_side = "left"
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True,
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.3,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    prompt_lengths = inputs["input_ids"].shape[1]
    results = []
    for out in output_ids:
        new_tokens = out[prompt_lengths:]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        results.append(text.strip())
    return results
