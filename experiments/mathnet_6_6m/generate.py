"""Generate and save reproducible qualitative samples from the best checkpoint."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import torch
from tokenizers import Tokenizer

from .config import DEFAULT_MODEL_CONFIG, ModelConfig
from .model import MathNetGPT
from .train import choose_device


PROMPTS = (
    "<|bos|>\n<|problem|>\nFind the sum of the first 10 positive integers.\n<|solution|>\n",
    "<|bos|>\n<|problem|>\nSolve for x: 2x + 3 = 11.\n<|solution|>\n",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, help="Defaults to run-dir/checkpoints/best.pt")
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    checkpoint_path = args.checkpoint or args.run_dir / "checkpoints" / "best.pt"
    tokenizer_path = args.run_dir / "tokenizer.json"
    if not checkpoint_path.exists() or not tokenizer_path.exists():
        raise FileNotFoundError("A best checkpoint and copied tokenizer.json are required.")
    torch.manual_seed(args.seed)
    device, _ = choose_device(args.device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    checkpoint_config = checkpoint.get("run_config", {}).get("model")
    model_config = DEFAULT_MODEL_CONFIG if checkpoint_config is None else ModelConfig(**checkpoint_config)
    model = MathNetGPT(model_config).to(device).eval()
    model.load_state_dict(checkpoint["model_state_dict"])
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    sections = [
        "# Generated samples\n",
        f"Generated at {datetime.now(timezone.utc).isoformat()} from `{checkpoint_path.name}`. "
        "These are qualitative samples, not correctness-scored solutions.\n",
    ]
    for index, prompt in enumerate(PROMPTS, start=1):
        prompt_ids = torch.tensor([tokenizer.encode(prompt).ids], dtype=torch.long, device=device)
        generated_ids = model.generate(
            prompt_ids,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
        )[0].tolist()
        generated = tokenizer.decode(generated_ids, skip_special_tokens=False)
        sections.extend([f"## Sample {index}\n", "```text\n", generated, "\n```\n"])
    (args.run_dir / "generation_samples.md").write_text("".join(sections), encoding="utf-8")


if __name__ == "__main__":
    main(parse_args())
