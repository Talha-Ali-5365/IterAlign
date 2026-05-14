import os
import pickle
import time
import gc
import json
import torch
import yaml
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.model import ensure_model_downloaded, load_model_and_tokenizer, generate_responses
from src.dataset import load_red_teaming_data, load_bad_cases
from src.utils import (
    setup_oracle_client,
    oracle_evaluate,
    propose_constitution,
    run_sft,
    save_checkpoint,
    load_checkpoint,
)


def main():
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    model_cfg = config["model"]
    ds_cfg = config["dataset"]
    train_cfg = config["training"]
    oracle_cfg = config["oracle"]

    BATCH_SIZE = train_cfg.get("batch_size", 20)

    model_dir = ensure_model_downloaded(model_cfg["repo"], model_cfg["local_dir"])

    client = setup_oracle_client(oracle_cfg.get("base_url"))
    ORACLE_MODEL = oracle_cfg["model"]

    OUTPUT_DIR = train_cfg.get("output_dir", "./output")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    bad_cases_all_sorted = load_bad_cases(ds_cfg["data_dir"])
    initial_red_teaming_data_all = load_red_teaming_data(
        ds_cfg["name"],
        data_dir=ds_cfg["data_dir"],
        max_items=ds_cfg["max_items"],
    )

    length_rt_data_all = len(initial_red_teaming_data_all)
    total_batches = (length_rt_data_all + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"Dataset: {ds_cfg['name']} | Prompts: {length_rt_data_all} | Batches: {total_batches}")
    print(f"Output dir: {OUTPUT_DIR}")

    state = load_checkpoint(OUTPUT_DIR)
    if state:
        start_i = state["next_i"]
        batch_id = state["batch_id"]
        current_model_path = state["current_model_path"]
        print(f"Resuming from prompt index {start_i}, batch_id={batch_id}, model={current_model_path}")
    else:
        start_i = 0
        batch_id = 0
        current_model_path = model_dir
        print("Starting fresh run.")

    remaining_batches = (length_rt_data_all - start_i + BATCH_SIZE - 1) // BATCH_SIZE
    outer_bar = tqdm(
        range(start_i, length_rt_data_all, BATCH_SIZE),
        total=remaining_batches,
        desc="Batches",
        unit="batch",
    )

    model, tokenizer = load_model_and_tokenizer(current_model_path)

    for i in outer_bar:
        batch_start = time.time()
        outer_bar.set_postfix(batch_id=batch_id, sft_runs=batch_id)

        print(f"\n{'='*60}")
        print(f"[Batch {batch_id}] prompts {i}–{min(i+BATCH_SIZE, length_rt_data_all)-1} / {length_rt_data_all-1}")
        print(f"{'='*60}")

        prompts = initial_red_teaming_data_all[i : i + BATCH_SIZE]

        print("Generating responses...")
        generated_texts = generate_responses(model, tokenizer, prompts)

        print("Evaluating responses with oracle (parallel)...")
        oracle_evals = [None] * len(generated_texts)
        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_idx = {
                executor.submit(oracle_evaluate, client, ORACLE_MODEL, bad_cases_all_sorted, text): idx
                for idx, text in enumerate(generated_texts)
            }
            for future in tqdm(as_completed(future_to_idx), total=len(generated_texts), desc="  Oracle eval", leave=False):
                idx = future_to_idx[future]
                oracle_evals[idx] = future.result()

        print("\n--- Sample responses (first 5) ---")
        for k in range(min(5, len(generated_texts))):
            verdict = "Positive" if "Negative" not in oracle_evals[k] else "Negative"
            print(f"[{k}] {verdict}")
            print(f"  Response : {generated_texts[k][:120].replace(chr(10), ' ')}...")
            print(f"  Oracle   : {oracle_evals[k][:100].replace(chr(10), ' ')}")
        print("---")

        neg_prompts = []
        neg_responses = []
        for j, eval_result in enumerate(oracle_evals):
            if "Negative" in eval_result:
                neg_responses.append(generated_texts[j])
                neg_prompts.append(prompts[j])

        print(f"Negative responses: {len(neg_responses)} / {len(generated_texts)}")

        if len(neg_responses) == 0:
            elapsed = time.time() - batch_start
            save_checkpoint(OUTPUT_DIR, i + BATCH_SIZE, batch_id, current_model_path)
            print(f"No negative responses in batch {batch_id}. Skipping fine-tuning. ({elapsed:.1f}s)")
            continue

        print("Proposing constitution...")
        constitution_response = propose_constitution(client, ORACLE_MODEL, bad_cases_all_sorted, neg_responses)
        print(f"Constitution:\n{constitution_response}")

        top_constitution = constitution_response.split("\n\n")[-1]

        with open(f"{OUTPUT_DIR}/constitution_batch_{batch_id}.txt", "w") as f:
            f.write(top_constitution)

        constitution_prompts = [
            "\n\n".join(text.split("\n\n")[:-1]) + " " + top_constitution + "\n\n" + text.split("\n\n")[-1]
            for text in neg_prompts
        ]

        print(f"Generating {len(constitution_prompts)} revised responses with constitution...")
        revised_responses = generate_responses(model, tokenizer, constitution_prompts)

        del model
        gc.collect()
        torch.cuda.empty_cache()

        with open(f"{OUTPUT_DIR}/neg_prompts_batch_{batch_id}.pkl", "wb") as f:
            pickle.dump(neg_prompts, f)

        sft_data = []
        for k, neg_prompt in enumerate(neg_prompts):
            sft_data.append({
                "id": k,
                "conversations": [
                    {"from": "human", "value": neg_prompt},
                    {"from": "gpt", "value": revised_responses[k]},
                ],
            })

        sft_data_path = f"{OUTPUT_DIR}/SFT_data_batch_{batch_id}.json"
        with open(sft_data_path, "w") as f:
            json.dump(sft_data, f)

        print(f"SFT data saved: {len(sft_data)} examples -> {sft_data_path}")

        sft_output = f"{OUTPUT_DIR}/sft_batch_{batch_id}"
        run_sft(current_model_path, sft_data_path, sft_output, config)

        current_model_path = sft_output
        model, tokenizer = load_model_and_tokenizer(current_model_path)
        batch_id += 1
        save_checkpoint(OUTPUT_DIR, i + BATCH_SIZE, batch_id, current_model_path)

        elapsed = time.time() - batch_start
        print(f"[Batch {batch_id - 1}] complete in {elapsed:.1f}s")

    del model
    gc.collect()
    torch.cuda.empty_cache()
    print(f"\nIterAlign training complete. Total SFT runs: {batch_id}")


if __name__ == "__main__":
    main()
