### Binary classification quality evaluation (medical reasonableness, no diagnosis leakage) ###

import os
import json
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed

# === API Settings ===
OPENAI_API_KEY = "<API_KEY>"
OPENAI_API_BASE = ""

client = OpenAI(
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_API_BASE,
    max_retries=5,
)

# === Binary evaluation function ===
def evaluate_quality(item):
    dialogue = item.get("chat_history")
    diagnosis = item.get("name", "")
    item_id = item.get("id", f"no-id-{hash(dialogue)}")

    if not dialogue or not dialogue.strip():
        return {"id": item_id, "error": "Empty dialogue", "name": diagnosis}

    evaluation_prompt = f"""Please strictly judge whether the "patient statements" and "test results" in the following dialogue meet two conditions:

1. Medical reasonableness: The patient's statements and test results match the typical clinical presentation or medical logic of the disease "{diagnosis}". The patient's chief complaint, symptoms, signs, and test results should be consistent with the known features of the disease and contain no obvious contradictions or unreasonable content. The doctor may ask some irrelevant questions or order unnecessary tests; you only need to judge whether the responses match the disease characteristics, not the doctor's rationality.
2. No diagnosis leakage: Before the doctor infers the diagnosis, did the patient or test results directly reveal the confirmed diagnosis?
Please distinguish between reasonable statements/results and direct leakage:
- Reasonable statements/results: consistent with disease features but do not directly state the diagnosis, e.g., "I have been very thirsty lately and drinking water doesn't help" (a symptom of diabetes).
- Direct leakage: explicitly states the diagnosis, e.g., "I have diabetes."
- If the doctor orders tests whose results are expected to contain the diagnosis (e.g., pathology results, imaging diagnosis), it is considered reasonable and not leakage.

If correct, output "1"; otherwise output "0".

📌 Output requirements:
- Output only valid JSON, no extra characters:
{{
  "Medical Reasonableness": 0 or 1,
  "No Diagnosis Leakage": 0 or 1
}}

Dialogue:
{dialogue}

Correct diagnosis: {diagnosis}
"""

    messages = [
        {"role": "system", "content": "You are a senior clinical expert."},
        {"role": "user", "content": evaluation_prompt}
    ]

    try:
        completion = client.chat.completions.create(
            model="gpt-5",
            messages=messages,
            temperature=0.0,  # Reduce randomness
        )
        output = completion.choices[0].message.content.strip()
    except Exception as api_error:
        return {
            "id": item_id,
            "error": f"API call failed: {str(api_error)}",
            "name": diagnosis
        }

    # Clean potential wrappers
    output = output.replace("```json", "").replace("```", "").strip()

    try:
        result = json.loads(output)
        # Force to integer 0 or 1
        result["Medical Reasonableness"] = 1 if result.get("Medical Reasonableness") in [1, "1", True, "True"] else 0
        result["No Diagnosis Leakage"] = 1 if result.get("No Diagnosis Leakage") in [1, "1", True, "True"] else 0
        return {
            "id": item_id,
            "name": diagnosis,
            "scores": result
        }
    except Exception as e:
        return {
            "id": item_id,
            "error": f"JSON parse failed or invalid format: {str(e)}",
            "name": diagnosis,
            "raw_output": output
        }


# === Batch processing main function ===
def process_model_files(model_files):
    all_summary = {}

    for input_file in model_files:
        if not os.path.exists(input_file):
            print(f"⚠️ File does not exist: {input_file}")
            continue

        print(f"🚀 Processing: {input_file}")
        with open(input_file, "r", encoding="utf-8") as f:
            data = [json.loads(line) for line in f]

        input_file_clean = input_file.replace(".jsonl", "")
        scored_output_path = f"patient_score/new/{input_file_clean}_binary_eval.jsonl"
        failed_output_path = f"patient_score/new/{input_file_clean}_binary_eval_failed.jsonl"

        os.makedirs("score", exist_ok=True)

        valid_scores = []
        failed = []

        with open(scored_output_path, "w", encoding="utf-8") as fout, \
             open(failed_output_path, "w", encoding="utf-8") as ferr:

            with ThreadPoolExecutor(max_workers=50) as executor:
                futures = [executor.submit(evaluate_quality, item) for item in data]

                for i, future in enumerate(as_completed(futures), 1):
                    result = future.result()

                    if "error" in result:
                        failed.append(result)
                        ferr.write(json.dumps(result, ensure_ascii=False) + "\n")
                    else:
                        fout.write(json.dumps(result, ensure_ascii=False) + "\n")
                        valid_scores.append(result["scores"])

                    print(f"[{i}/{len(data)}] Completed: {result.get('id', 'unknown')}")

        # === Summary statistics ===
        if valid_scores:
            total = len(valid_scores)
            avg_medical = sum(r["Medical Reasonableness"] for r in valid_scores) / total
            avg_no_leak = sum(r["No Diagnosis Leakage"] for r in valid_scores) / total

            summary = {
                "Medical Reasonableness Pass Rate": round(avg_medical, 4),
                "No Diagnosis Leakage Pass Rate": round(avg_no_leak, 4),
                "Sample Count": total
            }
        else:
            summary = {
                "Medical Reasonableness Pass Rate": None,
                "No Diagnosis Leakage Pass Rate": None,
                "Sample Count": 0
            }

        all_summary[os.path.basename(input_file_clean)] = summary

        print(f"✅ Completed: {input_file}")
        print(f" - Valid results: {scored_output_path}")
        print(f" - Failed items: {len(failed)}")

    # Save overall summary
    with open("binary_evaluation_summary.json", "w", encoding="utf-8") as f:
        json.dump(all_summary, f, indent=2, ensure_ascii=False)

    print("🎉 Evaluation completed, summary saved as binary_evaluation_summary.json")


# === Entry point ===
if __name__ == "__main__":
    model_files = [
        "result_DeepSeek-R1.jsonl",
        "result_4o-new.jsonl",
        "result_gemini-2.5-pro.jsonl",
        "result_gpt-5-mini.jsonl",
        "result_gpt-5-nano.jsonl",
        "result_qwen2.5-7b-new.jsonl",
        "result_qwen3-235b-a22b.jsonl"
        # Add more files if needed
    ]

    if model_files:
        process_model_files(model_files)
    else:
        print("⚠️ No files found to process")
