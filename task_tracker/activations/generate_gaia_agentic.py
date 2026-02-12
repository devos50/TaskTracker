import argparse
import json
import logging
import os
from typing import Dict, List, Optional

from datasets import load_dataset

from task_tracker.config.models import cache_dir, models
from task_tracker.utils.activations import process_texts_in_batches_agentic
from task_tracker.utils.model import load_model


def _pick_text_field(sample: Dict, preferred_fields: List[str]) -> str:
    for field in preferred_fields:
        if field in sample and isinstance(sample[field], str) and sample[field].strip():
            return field
    for key, value in sample.items():
        if isinstance(value, str) and value.strip():
            return key
    raise ValueError("No string field found in GAIA sample to use as task text.")


def _build_agentic_items(
    dataset, text_field: Optional[str] = None, max_items: Optional[int] = None
) -> List[Dict[str, str]]:
    preferred_fields = ["question", "task", "prompt", "query", "instruction"]
    if text_field is None:
        text_field = _pick_text_field(dataset[0], preferred_fields)
        logging.info(f"Using text field '{text_field}' for GAIA items.")

    items: List[Dict[str, str]] = []
    limit = max_items if max_items and max_items > 0 else len(dataset)
    for i in range(limit):
        text = dataset[i].get(text_field, "")
        items.append({"task": text, "step": text})
    return items


def main():
    parser = argparse.ArgumentParser(
        description="Generate activations for GAIA agentic tasks."
    )
    parser.add_argument("--model-name", default="phi3")
    parser.add_argument("--split", default="test")
    parser.add_argument("--max-items", type=int, default=0)
    parser.add_argument("--with-priming", action="store_true")
    parser.add_argument("--text-field", default=None)
    parser.add_argument(
        "--output-json",
        default="task_tracker/dataset_creation/gaia/gaia_{split}.json",
    )
    parser.add_argument(
        "--sub-dir-name",
        default="gaia_test",
        help="Subdirectory under the model output_dir to store activations.",
    )
    args = parser.parse_args()

    model = models[args.model_name]

    loaded_model = load_model(
        model.name, cache_dir=cache_dir, torch_dtype=model.torch_dtype
    )
    model.tokenizer = loaded_model["tokenizer"]
    model.model = loaded_model["model"]
    model.model.eval()

    dataset = load_dataset("gaia-benchmark/GAIA", split=args.split)

    items = _build_agentic_items(
        dataset, text_field=args.text_field, max_items=args.max_items
    )

    output_json = args.output_json.format(split=args.split)
    output_dir = os.path.dirname(output_json)
    os.makedirs(output_dir, exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(items, f, indent=2)

    process_texts_in_batches_agentic(
        dataset_subset=items,
        model=model,
        data_type=args.split,
        sub_dir_name=args.sub_dir_name,
        with_priming=args.with_priming,
        task_key="task",
        step_key="step",
    )


if __name__ == "__main__":
    main()
