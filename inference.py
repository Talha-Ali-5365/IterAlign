import os
import yaml

from src.model import ensure_model_downloaded, load_model_and_tokenizer, generate_responses


def main():
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    model_cfg = config["model"]
    local_dir = ensure_model_downloaded(model_cfg["repo"], model_cfg["local_dir"])

    model, tokenizer = load_model_and_tokenizer(local_dir)

    example_prompts = [
        "Explain the importance of data privacy in simple terms.",
        "What are some effective ways to reduce stress?",
        "How do I make a password more secure?",
    ]

    print("\n" + "=" * 60)
    print("Generating responses for example prompts...")
    print("=" * 60)
    responses = generate_responses(model, tokenizer, example_prompts, max_new_tokens=200)

    for i, (prompt, response) in enumerate(zip(example_prompts, responses)):
        print(f"\n--- Prompt {i + 1} ---")
        print(f"Q: {prompt}")
        print(f"A: {response}")

    print("\nInference complete.")


if __name__ == "__main__":
    main()
