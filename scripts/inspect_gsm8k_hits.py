"""Replay a deterministic GSM8K evaluation and save only correct responses."""

import argparse
import json
import os

from nanochat.checkpoint_manager import load_model
from nanochat.common import autodetect_device_type, compute_cleanup, compute_init
from nanochat.engine import Engine
from tasks.gsm8k import GSM8K, extract_answer


def main():
    parser = argparse.ArgumentParser(description="Save correct GSM8K generations.")
    parser.add_argument("--source", choices=["sft", "rl"], default="sft")
    parser.add_argument("--model-tag", required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--device-type", choices=["cuda", "cpu", "mps"], default="")
    parser.add_argument("--max-problems", type=int, default=128)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    device_type = autodetect_device_type() if args.device_type == "" else args.device_type
    _, rank, _, world_size, device = compute_init(device_type)
    if world_size != 1:
        raise ValueError("inspect_gsm8k_hits currently requires a single process")

    model, tokenizer, _ = load_model(
        args.source,
        device,
        phase="eval",
        model_tag=args.model_tag,
        step=args.step,
    )
    engine = Engine(model, tokenizer)
    task = GSM8K(subset="main", split="test")
    total = min(len(task), args.max_problems)
    hits = []

    for index in range(total):
        conversation = task[index]
        encoded_prompt = tokenizer.render_for_completion(conversation)
        results, _ = engine.generate_batch(
            encoded_prompt,
            num_samples=1,
            max_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
        )
        response = tokenizer.decode(results[0][len(encoded_prompt):])
        if task.evaluate(conversation, response):
            raw_row = task.ds[index]
            hit = {
                "evaluation_index": index,
                "question": raw_row["question"],
                "reference_solution": raw_row["answer"],
                "reference_answer": extract_answer(raw_row["answer"]),
                "model_response": response,
                "predicted_answer": extract_answer(response),
            }
            hits.append(hit)
            print(json.dumps(hit, ensure_ascii=False, indent=2), flush=True)
        print(f"progress={index + 1}/{total} hits={len(hits)}", flush=True)

    output_path = os.path.abspath(os.path.expanduser(args.output))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        for hit in hits:
            handle.write(json.dumps(hit, ensure_ascii=False) + "\n")
    print(f"saved {len(hits)} hits to {output_path}")
    compute_cleanup()


if __name__ == "__main__":
    main()
