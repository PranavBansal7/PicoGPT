"""Run a friendly local terminal demo of a trained PicoGPT MathNet checkpoint.

This is an interactive problem-to-solution demo, rather than a chat-tuned
assistant. Each submitted question is generated independently because the
model was trained on individual MathNet problem → solution documents.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .chat import load_chat_model, respond


HELP_TEXT = """\
Commands:
  :help       Show these commands.
  :settings   Show the active generation settings.
  :new        Start a fresh displayed session.
  :quit       Exit the demo.

Ask one self-contained math question at a time. For a follow-up, restate the
relevant information: the model does not receive an implicit conversation
history because its training data uses individual problem → solution examples.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Directory containing tokenizer.json and checkpoints/best.pt.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Defaults to run-dir/checkpoints/best.pt.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="Maximum generated tokens per answer (default: 128).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature; lower is less random (default: 0.7).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=40,
        help="Keep the top K token candidates; use 1 for repeatable greedy output.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help="Use auto to prefer CUDA when it is available.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.max_new_tokens < 1:
        raise ValueError("--max-new-tokens must be positive")
    if args.temperature <= 0:
        raise ValueError("--temperature must be positive")
    if args.top_k < 1:
        raise ValueError("--top-k must be at least 1")


def settings_text(args: argparse.Namespace, device: object) -> str:
    checkpoint = args.checkpoint or args.run_dir / "checkpoints" / "best.pt"
    return (
        f"run directory: {args.run_dir}\n"
        f"checkpoint: {checkpoint}\n"
        f"device: {device}\n"
        f"max new tokens: {args.max_new_tokens}\n"
        f"temperature: {args.temperature}\n"
        f"top-k: {args.top_k}"
    )


def main(args: argparse.Namespace) -> None:
    validate_args(args)
    print("Loading PicoGPT MathNet checkpoint...")
    model, tokenizer, eos_token_id, device = load_chat_model(
        args.run_dir, args.checkpoint, args.device
    )
    print(
        "\nPicoGPT MathNet local demo\n"
        "Enter a self-contained math question. Type :help for commands.\n"
        "Note: answers are sampled continuations and require verification.\n"
    )
    print(settings_text(args, device))

    turn = 1
    while True:
        try:
            question = input(f"\nYou [{turn}]> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            return

        command = question.lower()
        if command in {":quit", ":exit", "quit", "exit"}:
            print("Goodbye.")
            return
        if command == ":help":
            print("\n" + HELP_TEXT)
            continue
        if command == ":settings":
            print("\n" + settings_text(args, device))
            continue
        if command in {":new", ":reset"}:
            turn = 1
            print("\nStarted a new displayed session.")
            continue
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
                top_k=args.top_k,
            )
        except ValueError as error:
            print(f"\nPicoGPT> {error}")
            continue
        except RuntimeError as error:
            print(
                "\nPicoGPT> Generation failed. If CUDA is unavailable or out of memory, "
                "restart with --device cpu.\n"
                f"Details: {error}"
            )
            continue

        print("\nPicoGPT> " + (answer or "[generated EOS without a textual solution]"))
        turn += 1


if __name__ == "__main__":
    main(parse_args())
