"""Interact with a trained PicoGPT MathNet checkpoint from a terminal or Colab.

The model was trained on problem → solution documents, not dialogue data. This
is therefore a mathematical-question interface, not a general ChatGPT-style
assistant; its replies are sampled continuations and can be wrong.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from tokenizers import Tokenizer

from .config import DEFAULT_MODEL_CONFIG, ModelConfig
from .model import MathNetGPT
from .train import choose_device


def load_chat_model(
    run_dir: Path, checkpoint_path: Path | None, device_name: str
) -> tuple[MathNetGPT, Tokenizer, int, torch.device]:
    checkpoint_path = checkpoint_path or run_dir / "checkpoints" / "best.pt"
    tokenizer_path = run_dir / "tokenizer.json"
    if not checkpoint_path.exists() or not tokenizer_path.exists():
        raise FileNotFoundError(
            "A trained checkpoint and its tokenizer.json are required. "
            "Run training in Colab before starting chat."
        )
    device, _ = choose_device(device_name)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    raw_config = checkpoint.get("run_config", {}).get("model")
    config = DEFAULT_MODEL_CONFIG if raw_config is None else ModelConfig(**raw_config)
    model = MathNetGPT(config).to(device).eval()
    model.load_state_dict(checkpoint["model_state_dict"])
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    eos_token_id = tokenizer.token_to_id("<|eos|>")
    if eos_token_id is None:
        raise ValueError("The saved tokenizer has no <|eos|> token.")
    return model, tokenizer, eos_token_id, device


def make_prompt(question: str) -> str:
    return "<|bos|>\n<|problem|>\n" + question.strip() + "\n<|solution|>\n"


@torch.no_grad()
def respond(
    model: MathNetGPT,
    tokenizer: Tokenizer,
    eos_token_id: int,
    device: torch.device,
    question: str,
    *,
    max_new_tokens: int,
    temperature: float,
    top_k: int | None,
) -> str:
    prompt_ids = tokenizer.encode(make_prompt(question)).ids
    if len(prompt_ids) >= model.config.block_size:
        raise ValueError(
            f"Question is {len(prompt_ids)} tokens; it must be shorter than "
            f"the {model.config.block_size}-token context window."
        )
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    generated = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        eos_token_id=eos_token_id,
    )[0].tolist()
    answer_ids = generated[len(prompt_ids) :]
    if eos_token_id in answer_ids:
        answer_ids = answer_ids[: answer_ids.index(eos_token_id)]
    return tokenizer.decode(answer_ids, skip_special_tokens=True).strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, help="Defaults to run-dir/checkpoints/best.pt")
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=40, help="Use 0 to disable top-k filtering.")
    parser.add_argument("--question", help="Answer one question and exit; useful in Colab cells.")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    if args.max_new_tokens < 1:
        raise ValueError("max-new-tokens must be positive")
    if args.temperature <= 0:
        raise ValueError("temperature must be positive")
    if args.top_k < 0:
        raise ValueError("top-k must be non-negative")
    model, tokenizer, eos_token_id, device = load_chat_model(args.run_dir, args.checkpoint, args.device)
    top_k = None if args.top_k == 0 else args.top_k
    if args.question is not None:
        print(
            respond(
                model,
                tokenizer,
                eos_token_id,
                device,
                args.question,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=top_k,
            )
        )
        return
    print("PicoGPT MathNet chat — enter a math question. Type :quit to exit.")
    while True:
        try:
            question = input("\nYou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            return
        if question.lower() in {":quit", ":exit", "quit", "exit"}:
            print("Goodbye.")
            return
        if not question:
            continue
        try:
            answer = respond(
                model,
                tokenizer,
                eos_token_id,
                device,
                question,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=top_k,
            )
            print("\nPicoGPT> " + (answer or "[generated EOS without a textual solution]"))
        except ValueError as error:
            print(f"\nPicoGPT> {error}")


if __name__ == "__main__":
    main(parse_args())
