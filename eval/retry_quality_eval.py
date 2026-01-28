import os
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

from quality_eval import evaluate_quality  # Replace with the actual module name or paste evaluate_quality directly

# Max concurrency
MAX_WORKERS = 10

# Raw jsonl files to process in the current directory
model_files = [
    "result_gemini-2.5-pro.jsonl",
    "result_deepseek-r1.jsonl",
    "result_deepseek-v3.jsonl",
    "result_gpt-5-mini.jsonl",
    "result_gpt-5-nano.jsonl",
    "result_qwen3-8b-new.jsonl",
    "result_gemini-2.5-pro.jsonl",
    "result_DeepSeek-R1.jsonl",
    "result_deepseek-v3.jsonl",
    "result_qwen2.5-7b-new.jsonl",
    "result_qwen3-235b-a22b.jsonl",
    "result_qwen3-next-80b-a3b-instruct.jsonl"
    "result_llama-4-scout.jsonl",
    "result_meta-llama/llama-4-maverick.jsonl",
    "result_claude-sonnet-4-20250514.jsonl"   
    # Add more files if needed 
]

for raw_file in model_files:
    if not os.path.exists(raw_file):
        print(f"⚠️ Raw file does not exist: {raw_file}")
        continue

    print(f"🔍 Checking missing scored items: {raw_file}")

    # Generate the corresponding scored file path
    scored_file = f"score/{raw_file.replace('.jsonl', '')}_quality_evaluation_scored.jsonl"
    os.makedirs(os.path.dirname(scored_file), exist_ok=True)

    # Read raw data
    with open(raw_file, "r", encoding="utf-8") as f:
        raw_items = [json.loads(line) for line in f]

    # Read names of already scored items
    scored_names = set()
    if os.path.exists(scored_file):
        with open(scored_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    item = json.loads(line)
                    if "name" in item:
                        scored_names.add(item["name"])
                except:
                    continue

    # Find missing items (present in raw file but absent in scored file by name)
    to_retry = [item for item in raw_items if item.get("name") not in scored_names]
    print(f" - Missing scored item count: {len(to_retry)}")

    if not to_retry:
        continue

    new_failed = []

    with open(scored_file, "a", encoding="utf-8") as fout:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(evaluate_quality, item): item for item in to_retry}

            for i, future in enumerate(as_completed(futures), 1):
                item = futures[future]
                try:
                    result = future.result()
                except Exception as e:
                    print(f"[{i}/{len(to_retry)}] evaluate_quality error: {item.get('name', 'unknown')} -> {e}")
                    new_failed.append(item)
                    continue

                if "error" in result:
                    print(f"[{i}/{len(to_retry)}] Still failed: {result.get('name', 'unknown')} -> {result['error']}")
                    new_failed.append(item)
                else:
                    fout.write(json.dumps(result, ensure_ascii=False) + "\n")
                    print(f"[{i}/{len(to_retry)}] Retry succeeded: {result.get('name', 'unknown')}")

    # Save failed items to a separate file (optional)
    if new_failed:
        failed_file = scored_file.replace("_scored.jsonl", "_retry_failed.jsonl")
        with open(failed_file, "w", encoding="utf-8") as ferr:
            for item in new_failed:
                ferr.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f" - Remaining failed items saved to: {failed_file} (total {len(new_failed)})")

    print(f"✅ Completed file: {raw_file}\n")
