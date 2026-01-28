### Quality scoring across 7 dimensions ###

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

# === Scoring dimensions and their maximum scores ===
SCORING_KEYS = {
    "Chief Complaint Exploration Depth": 10,
    "History Completeness": 10,
    "Evidence Chain Completeness": 20,
    "Test Appropriateness": 10,
    "Differential Diagnosis": 10,
    "Diagnostic Accuracy": 30,
    "Uncertainty Management": 10,
}

# Calculate maximum total score
TOTAL_MAX_SCORE = sum(SCORING_KEYS.values())

# === Evaluation function (enhanced error handling) ===
def evaluate_quality(item):
    dialogue = item.get("chat_history")
    final_answer = item.get("final_answer", "")
    diagnosis = item.get("name", "")
    item_id = item.get("id", f"no-id-{hash(dialogue)}")

    if not dialogue or not dialogue.strip():
        return {"id": item_id, "error": "Empty dialogue", "name": diagnosis}

    evaluation_prompt = f"""You are a senior clinical expert with more than 15 years of clinical experience in top-tier tertiary hospitals and a former member of a teaching assessment committee. You are appointed to conduct a strict professional evaluation of the following physician consultation record and clinical reasoning process. Your scoring will be used for medical quality audit and case review, and must align with clinical practice requirements.

Please score the consultation content across the following 7 dimensions. Each score must be an integer and based on clear clinical evidence. The scoring criteria must strictly follow the standards below, without loosening or subjective inference.

📌 Output format requirements:
- Output only standard JSON; the field order and structure must be exactly as follows, with no comments, no line breaks, and no extra text:
{{
  "Chief Complaint Exploration Depth": score,
  "History Completeness": score,
  "Evidence Chain Completeness": score,
  "Test Appropriateness": score,
  "Differential Diagnosis": score,
  "Diagnostic Accuracy": score,
  "Uncertainty Management": score
}}

📌 Scoring dimensions and thresholds (maximum scores are weighted by clinical importance):

1. [Chief Complaint Exploration Depth] (max 10)
   - 10: Structured collection of symptom features (onset time, nature, location, intensity, triggers, relieving factors, associated symptoms), and identifies at least one "red-flag sign" (e.g., chest pain with cold sweat, headache with altered consciousness).
   - 6: Covers basic symptom elements but does not systematically probe features or fails to identify red flags.
   - 4: Only records the patient's words, without clarifying vague descriptions (e.g., "stomach discomfort" without location/characterization).
   - 2: Chief complaint description is vague, missing key symptom dimensions.
   - 0: Fails to identify symptom features requiring emergency intervention (e.g., chest pain without asking about radiation, dyspnea without asking about rest status).
   - ▶ Deduction triggers: no proactive follow-up questions -> up to 3 points; no record of symptom duration or frequency -> up to 2 points.

2. [History Completeness] (max 10)
   History includes: present illness, past medical history, medication history, allergy history, family history, social history. Each item mentioned earns 2 points, up to 10 points.

3. [Evidence Chain Completeness] (max 20)
   - 20: Each clinical judgment (e.g., "consider infection," "leaning cardiac") is supported by corresponding symptoms, signs, or test results; the reasoning chain is complete with no leaps.
   - 15: One judgment has weak supporting evidence (e.g., diagnosing pneumonia without fever or lung auscultation findings).
   - 10: A key diagnostic hypothesis lacks direct evidence (e.g., diagnosing cholecystitis without Murphy sign).
   - 5: Subjective inference exists (e.g., "patient is anxious" without HAMA score or behavioral description).
   - ≤2: Multiple conclusions lack objective basis, or uses non-evidence expressions like "experience judgment" or "feels like."
   - ▶ Deduction triggers: uses "possibly/maybe" without labeling uncertainty -> up to 3 points; diagnostic premise contradicts records -> 0 points.

4. [Test Appropriateness] (max 10)
   - 10: Tests precisely match the differential diagnosis, follow clinical pathway guidelines, no missing core tests, no unnecessary excessive tests, and test indications are clearly documented.
   - 8: One test indication is unclear, or one low-priority test is delayed (e.g., typical abdominal pain without prompt amylase testing).
   - 6: Obvious overtesting (e.g., MRI for a young patient with headache without indication) or omission of high-risk screening (e.g., reproductive-age woman with abdominal pain without a pregnancy test).
   - 4: Tests are weakly related to the chief complaint, or the clinical purpose is not explained.
   - ≤2: The test set does not follow routine logic, or high-risk patients are not prioritized for key tests (e.g., chest pain without ECG).
   - ▶ Deduction triggers: test purpose not stated -> minus 1 point; critical patients not given core tests at the first visit -> up to 2 points.

5. [Differential Diagnosis] (max 10)
   - 10: Lists ≥3 reasonable differentials, including "high-fatal-but-treatable" diseases (e.g., ACS, pulmonary embolism, stroke, ectopic pregnancy), ordered by clinical probability, and explains exclusion or supporting evidence for each.
   - 8: Lists 3 differentials but without ordering or with insufficient exclusion rationale.
   - 6: Only 2 differentials, missing required critical conditions.
   - 3-5: Only 1 differential, or differentials are clearly unreasonable.
   - ≤3: No differential diagnosis, or misses required "red-flag diseases."
   - ▶ Deduction triggers: fails to consider the most dangerous diagnosis in the symptom spectrum (e.g., headache without considering subarachnoid hemorrhage) -> 0 points.

6. [Diagnostic Accuracy] (max 30)
   - 30: Final diagnosis is highly consistent with all clinical evidence, follows the latest guidelines, with no logical contradictions; if evidence is insufficient, explicitly label as "preliminary diagnosis" or "to be ruled out," and provide rationale.
   - 20-29: Diagnosis is correct but does not sufficiently state confidence, does not systematically rule out key differentials, or does not label "preliminary" when evidence is somewhat insufficient.
   - 15-20: Diagnosis is largely correct but misses important comorbidities or complications (e.g., pneumonia without mentioning pleural effusion, diabetes without mentioning ketosis tendency), or some inferences lack direct evidence.
   - 10-15: Diagnosis direction is partially wrong or vague (e.g., cholecystitis misjudged as gastritis), but no missed high-risk disease and no obvious clinical risk.
   - 5-9: Diagnosis contradicts key positive signs or test results (e.g., ECG suggests MI yet diagnosed as gastritis), or ignores red-flag signs that must be addressed.
   - 0-4: Diagnosis is seriously wrong and may cause life-threatening or irreversible harm (e.g., aortic dissection misdiagnosed as muscle strain, ectopic pregnancy misdiagnosed as menstrual disorder).
   - ▶ Deduction triggers:
     - Diagnosis contradicts objective records -> directly ≤4 points;
     - Insufficient evidence but no "preliminary diagnosis" label -> up to 25 points;
     - Misses high-risk diseases that must be ruled out (e.g., chest pain without considering ACS) -> up to 17 points;
     - Uses vague diagnosis to mask uncertainty (e.g., "maybe XX" without a validation plan) -> up to 24 points.

7. [Uncertainty Management] (max 10)
   - 10: Clearly states sources of diagnostic or prognostic uncertainty, sets a specific verification plan (e.g., "follow up within 72 hours," "escalate imaging if no improvement"), and records risk communication with the patient.
   - 7: Mentions uncertainty, has a follow-up plan but without a quantified timeline or verification method.
   - 5-6: Uses only vague terms like "observe" or "follow-up," with no concrete action items.
   - 3-4: Uses absolute language to mask uncertainty (e.g., "definitely not cancer," "no problem").
   - ≤2: No mention of uncertainty, or gives misleading reassurance to the patient.
   - ▶ Deduction triggers: no record of risk disclosure or informed process -> up to 3 points; high-risk patients without a clear follow-up mechanism -> up to 4 points.

⚠️ Review principles reiterated:
- All scores must be based on **verifiable text records**; do not assume "the doctor may have done it but didn't write it."
- High-weight dimensions (Diagnostic Accuracy, Differential Diagnosis, Uncertainty Management) use a "defect-sensitive" scoring method — missing or incorrect key items will cause sharp score drops.
- As a review expert, your scores will enter physician competency records and medical safety databases; you must be responsible for the clinical rationality and legal rigor of your scoring.

Please evaluate the following consultation record according to the above standards:
Consultation record: {dialogue}

The model's diagnosis is:
Correct answer: {diagnosis}
Please provide your answer; do not include anything other than JSON scoring.
Please strictly follow the above standards and output pure JSON with no extra text."""
    # print(evaluation_prompt)
    messages = [
        {"role": "system", "content": "You are a clinical evaluation expert tasked with professionally scoring consultation records."},
        {"role": "user", "content": evaluation_prompt}
    ]

    try:
        completion = client.chat.completions.create(
            model="gpt-5",
            messages=messages,
            # temperature=0.1,
        )
        output = completion.choices[0].message.content.strip()
        print(output)
    except Exception as api_error:
        return {
            "id": item_id,
            "error": f"API call failed: {str(api_error)}",
            "name": diagnosis
        }

    if not output:
        return {
            "id": item_id,
            "error": "Empty LLM response",
            "name": diagnosis
        }

    output = output.replace("```json", "").replace("```", "").strip()

    if not output:
        return {
            "id": item_id,
            "error": "Empty after cleaning Markdown wrappers",
            "name": diagnosis
        }

    try:
        evaluation_json = json.loads(output)
    except json.JSONDecodeError as e:
        return {
            "id": item_id,
            "error": f"JSON decode failed: {str(e)}",
            "raw_output": output,
            "name": diagnosis
        }

    score_sum = 0
    score_dict = {}

    for key, max_score in SCORING_KEYS.items():
        entry = evaluation_json.get(key)
        if entry is not None:
            try:
                score = int(entry)
                if 0 <= score <= max_score:
                    score_dict[key] = score
                    score_sum += score
                else:
                    score_dict[key] = 0
            except (ValueError, TypeError):
                score_dict[key] = 0
        else:
            score_dict[key] = None

    score_dict["Total Score"] = score_sum

    return {
        "id": item_id,
        "name": diagnosis,
        "scores": score_dict,
        "llm_evaluation": evaluation_json,
        "raw_output": output,
    }


# === Main function: batch process multiple model files ===
def process_model_files(model_files):
    all_summary = {}

    for input_file in model_files:
        if not os.path.exists(input_file):
            print(f"⚠️ File does not exist: {input_file}")
            continue

        print(f"🚀 Processing model file: {input_file}")
        with open(input_file, "r", encoding="utf-8") as f:
            data = [json.loads(line) for line in f]

        input_file_clean = input_file.replace(".jsonl", "")
        scored_output_path = f"score/{input_file_clean}_quality_evaluation_scored.jsonl"
        failed_output_path = f"score/{input_file_clean}_quality_evaluation_failed.jsonl"

        summary = {key: [] for key in SCORING_KEYS}
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
                        for key in SCORING_KEYS:
                            score = result["scores"].get(key)
                            if score is not None:
                                summary[key].append(score)

                    print(f"[{i}/{len(data)}] Completed: {result.get('id', 'unknown')}")

        # === Summary statistics ===
        total_scores = []
        if os.path.exists(scored_output_path):
            with open(scored_output_path, "r", encoding="utf-8") as f:
                for line in f:
                    item = json.loads(line)
                    if "scores" in item and "Total Score" in item["scores"]:
                        total_scores.append(item["scores"]["Total Score"])

        # Compute per-dimension statistics
        summary_result = {}
        for key, vals in summary.items():
            if vals:
                avg_raw = sum(vals) / len(vals)
                max_score = SCORING_KEYS[key]
                avg_normalized = (avg_raw / max_score) * 100
                summary_result[key] = {
                    "Raw Average Score": round(avg_raw, 2),
                    "Normalized Average (100)": round(avg_normalized, 2),
                    "Sample Count": len(vals),
                    "Max Score": max_score
                }
            else:
                summary_result[key] = {
                    "Raw Average Score": None,
                    "Normalized Average (100)": None,
                    "Sample Count": 0,
                    "Max Score": SCORING_KEYS[key]
                }

        # Compute total score statistics
        if total_scores:
            total_avg_raw = sum(total_scores) / len(total_scores)
            total_avg_normalized = (total_avg_raw / TOTAL_MAX_SCORE) * 100
            summary_result["Total Score"] = {
                "Raw Average Score": round(total_avg_raw, 2),
                "Normalized Average (100)": round(total_avg_normalized, 2),
                "Sample Count": len(total_scores),
                "Max Score": TOTAL_MAX_SCORE
            }
        else:
            summary_result["Total Score"] = {
                "Raw Average Score": None,
                "Normalized Average (100)": None,
                "Sample Count": 0,
                "Max Score": TOTAL_MAX_SCORE
            }

        model_name = os.path.basename(input_file_clean)
        all_summary[model_name] = summary_result

        # Save per-model detailed summary file (optional, can be commented out)
        # with open(f"{input_file_clean}_quality_summary.json", "w", encoding="utf-8") as f:
        #     json.dump(summary_result, f, indent=2, ensure_ascii=False)

        print(f"✅ Completed model: {input_file}")
        print(f" - Scored results: {scored_output_path}")
        print(f" - Failed records: {failed_output_path} (total {len(failed)})")

    # === Save all-model summary statistics to a single file ===
    with open("all_models_quality_summary.json", "w", encoding="utf-8") as f:
        json.dump(all_summary, f, indent=2, ensure_ascii=False)

    print("🎉 All model evaluations completed, summary saved as all_models_quality_summary.json")


# === Example: process multiple model files ===
if __name__ == "__main__":
    # Auto-scan all .jsonl files in the current directory (exclude result files)
    # model_files = [f for f in os.listdir(".") if f.endswith(".jsonl") and not f.endswith(("_scored.jsonl", "_failed.jsonl", "_summary.json"))]
    
    # Or manually specify a file list
    model_files = [
        "result_gemini-2.5-pro.jsonl",
        "result_deepseek-r1.jsonl",
        "result_deepseek-v3.jsonl",
        "result_doubao-1-5-pro-32k-250115.jsonl",
        "result_gpt-5-mini.jsonl",
        "result_gpt-5-nano.jsonl",
        "result_qwen2.5-7b-new.jsonl",
        "result_qwen3-235b-a22b.jsonl",
        "result_qwen3-next-80b-a3b-instruct.jsonl"
        "result_llama-4-scout.jsonl",
        "result_meta-llama/llama-4-maverick.jsonl"
        "result_claude-sonnet-4-20250514.jsonl"
        # Add more files as needed
    ]
    
    if model_files:
        process_model_files(model_files)
    else:
        print("⚠️ No model files found to process")
