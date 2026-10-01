"""Prepare a reproducible, text-only MathNet language-model corpus.

MathNet is multimodal.  This experiment deliberately trains a text GPT, so
inline image references become a single ``<|image|>`` token and image bytes are
not decoded or redistributed.  The public dataset exposes one ``train`` split;
this script makes a provenance-grouped internal train/validation/test split.
Those held-out partitions are *not* the official MathNet-Solve benchmark.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from array import array
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Optional

import numpy as np
from datasets import load_dataset
from tokenizers import ByteLevelBPETokenizer, Tokenizer

from .config import DEFAULT_MODEL_CONFIG


DATASET_ID = "ShadenA/MathNet"
DATASET_CONFIG = "all"
DATASET_REVISION = "33e6b3bc254e6f0f0c1479b4252f5e9dd551d56c"
SPECIAL_TOKENS = [
    "<|bos|>",
    "<|eos|>",
    "<|problem|>",
    "<|solution|>",
    "<|image|>",
]
IMAGE_MARKDOWN = re.compile(r"!\[[^\]]*\]\(attached_image_\d+\.png\)")


def _text(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""


def normalize_markdown(value: object) -> str:
    """Keep text/math markup while replacing unavailable embedded figures."""
    return IMAGE_MARKDOWN.sub("<|image|>", _text(value))


def format_document(row: dict[str, Any]) -> Optional[str]:
    """Convert one solved MathNet record into a causal-LM training document."""
    problem = normalize_markdown(row.get("problem_markdown"))
    raw_solutions = row.get("solutions_markdown")
    solutions = (
        [normalize_markdown(solution) for solution in raw_solutions]
        if isinstance(raw_solutions, (list, tuple))
        else []
    )
    solutions = [solution for solution in solutions if solution]
    if not problem or not solutions:
        return None
    return "\n".join(
        ["<|bos|>", "<|problem|>", problem, "<|solution|>", "\n\n".join(solutions), "<|eos|>"]
    )


def provenance_group(row: dict[str, Any]) -> str:
    """Choose the strongest available grouping key to reduce near-duplicate leakage."""
    booklet_source = _text(row.get("booklet_source"))
    if booklet_source:
        return f"booklet_source:{booklet_source}"
    country = _text(row.get("country"))
    competition = _text(row.get("competition"))
    if country or competition:
        return f"competition:{country}|{competition}"
    return f"id:{_text(row.get('id'))}"


def split_name(group: str, seed: int) -> str:
    """Map a provenance group deterministically to the 80/10/10 internal split."""
    digest = hashlib.sha256(f"{seed}:{group}".encode("utf-8")).digest()
    bucket = int.from_bytes(digest[:8], byteorder="big") % 100
    return "train" if bucket < 80 else "validation" if bucket < 90 else "test"


def _iter_documents(raw_dataset: Any, name: str, seed: int) -> Iterator[str]:
    for row in raw_dataset:
        if split_name(provenance_group(row), seed) != name:
            continue
        document = format_document(row)
        if document is not None:
            yield document


def _record_manifest(
    raw_dataset: Any, output_dir: Path, seed: int
) -> tuple[dict[str, int], Counter[str]]:
    """Persist record IDs and source groups without copying the licensed corpus."""
    counts: Counter[str] = Counter()
    provenance_types: Counter[str] = Counter()
    path = output_dir / "split_manifest.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for row in raw_dataset:
            document = format_document(row)
            if document is None:
                continue
            group = provenance_group(row)
            name = split_name(group, seed)
            counts[name] += 1
            provenance_types[group.split(":", 1)[0]] += 1
            handle.write(
                json.dumps(
                    {
                        "id": _text(row.get("id")),
                        "split": name,
                        "provenance_group": group,
                        "country": _text(row.get("country")),
                        "competition": _text(row.get("competition")),
                        "language": _text(row.get("language")),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    return dict(counts), provenance_types


def _write_token_buffer(
    tokenizer: Tokenizer, documents: Iterator[str], destination: Path
) -> tuple[int, int]:
    """Encode a split into compact uint16 IDs suitable for random window sampling."""
    ids = array("H")
    documents_written = 0
    for document in documents:
        encoded = tokenizer.encode(document).ids
        if any(token_id >= 2**16 for token_id in encoded):
            raise ValueError("uint16 storage requires a vocabulary smaller than 65,536")
        ids.extend(encoded)
        documents_written += 1
    np.frombuffer(ids, dtype=np.uint16).tofile(destination)
    return documents_written, len(ids)


def build_data(
    output_dir: Path,
    *,
    dataset_id: str = DATASET_ID,
    dataset_config: str = DATASET_CONFIG,
    revision: str = DATASET_REVISION,
    vocab_size: int = DEFAULT_MODEL_CONFIG.vocab_size,
    seed: int = 13_337,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    raw = load_dataset(dataset_id, dataset_config, split="train", revision=revision)
    # Avoid image decoding and explicitly record the observed schema, which can
    # differ from older MathNet card revisions.
    selected_columns = [
        name
        for name in (
            "id",
            "problem_markdown",
            "solutions_markdown",
            "country",
            "competition",
            "language",
            "booklet_source",
        )
        if name in raw.column_names
    ]
    raw = raw.select_columns(selected_columns)
    if not {"problem_markdown", "solutions_markdown"}.issubset(raw.column_names):
        raise RuntimeError(f"Unexpected MathNet schema: {raw.column_names}")

    split_counts, provenance_types = _record_manifest(raw, output_dir, seed)
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train_from_iterator(
        _iter_documents(raw, "train", seed),
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=SPECIAL_TOKENS,
        show_progress=True,
    )
    tokenizer_path = output_dir / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))
    frozen_tokenizer = Tokenizer.from_file(str(tokenizer_path))
    actual_vocab_size = frozen_tokenizer.get_vocab_size()
    if actual_vocab_size != vocab_size:
        raise RuntimeError(
            f"Tokenizer size {actual_vocab_size} differs from requested model vocabulary {vocab_size}"
        )

    token_counts: dict[str, int] = {}
    encoded_document_counts: dict[str, int] = {}
    for name in ("train", "validation", "test"):
        documents, tokens = _write_token_buffer(
            frozen_tokenizer, _iter_documents(raw, name, seed), output_dir / f"{name}.bin"
        )
        encoded_document_counts[name] = documents
        token_counts[name] = tokens
        if tokens <= DEFAULT_MODEL_CONFIG.block_size:
            raise RuntimeError(f"{name} split has too few tokens for context windows")

    tokenizer_sha256 = hashlib.sha256(tokenizer_path.read_bytes()).hexdigest()
    manifest_sha256 = hashlib.sha256((output_dir / "split_manifest.jsonl").read_bytes()).hexdigest()
    metadata: dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "id": dataset_id,
            "config": dataset_config,
            "revision": revision,
            "rows_loaded": len(raw),
            "observed_columns": raw.column_names,
            "fingerprint": getattr(raw, "_fingerprint", None),
            "modality_handling": "Attached image markdown is replaced with <|image|>; image bytes are excluded.",
            "public_split_note": "The public release has one train split; these are PicoGPT internal held-out splits.",
        },
        "split": {
            "method": "SHA-256 seeded provenance-group assignment, 80/10/10",
            "seed": seed,
            "record_counts": split_counts,
            "encoded_document_counts": encoded_document_counts,
            "token_counts": token_counts,
            "provenance_key_counts": dict(provenance_types),
            "manifest_sha256": manifest_sha256,
        },
        "tokenizer": {
            "kind": "ByteLevelBPETokenizer",
            "vocab_size": actual_vocab_size,
            "special_tokens": SPECIAL_TOKENS,
            "train_split_only": True,
            "sha256": tokenizer_sha256,
        },
    }
    with (output_dir / "data_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-id", default=DATASET_ID)
    parser.add_argument("--dataset-config", default=DATASET_CONFIG)
    parser.add_argument("--revision", default=DATASET_REVISION)
    parser.add_argument("--vocab-size", type=int, default=DEFAULT_MODEL_CONFIG.vocab_size)
    parser.add_argument("--seed", type=int, default=13_337)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = build_data(
        args.output_dir,
        dataset_id=args.dataset_id,
        dataset_config=args.dataset_config,
        revision=args.revision,
        vocab_size=args.vocab_size,
        seed=args.seed,
    )
    print(json.dumps(result, indent=2))
