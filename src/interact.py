

import json
import os
from patient import Patient
from doctor import Doctor
from assistant import Assistant
import argparse
import time
from tqdm import tqdm
from filelock import FileLock

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--model', default="qwen2.5-7b-instruct", type=str)
args = parser.parse_args()

def safe_update_jsonl(file_path, key_name, key_value, field_name, new_entry):
    lock_path = file_path + '.lock'
    with FileLock(lock_path):
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                entries = [json.loads(line) for line in f]
        else:
            entries = []

        found = False
        for entry in entries:
            if entry.get(key_name) == key_value:
                entry[field_name].append(new_entry)
                found = True
                break

        if not found:
            entries.append({
                key_name: key_value,
                field_name: [new_entry]
            })

        with open(file_path, 'w', encoding='utf-8') as f:
            for entry in entries:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')

def safe_append_jsonl(file_path, record):
    lock_path = file_path + '.lock'
    with FileLock(lock_path):
        with open(file_path, 'a', encoding='utf-8') as f:
            json.dump(record, f, ensure_ascii=False)
            f.write('\n')

def interact(patient, doctor, assistant):
    turns = 15
    chat_history = ""
    question = patient.generate_initial_question()
    chat_history += f"Patient: {question}\n"
    print(f"Patient: {question}")

    while not doctor.check_finish() and turns > 0:
        try:
            turns -= 1
            action = doctor.provide_diagnosis(chat_history)
            chat_history += f"Doctor: {action}\n"
            print(f"Doctor: {action}\n")
            print("-" * 50)

            if "[!Test!]" in action:
                exam = action.split("[!Test!]")[1].strip().strip("()")
                result = assistant.check_up(exam)
                chat_history += f"Test Result: {result}\n"
                print(f"Test Result: {result}\n")
                safe_update_jsonl('examination_history.jsonl', 'name', patient.disease_name, 'examination_history', {"Q": exam, "A": result})

            elif "[!Ask!]" in action:
                question = action.split("[!Ask!]")[1].strip().strip("()")
                answer = patient.answer_question(question)
                chat_history += f"Patient: {answer}\n"
                print(f"Patient: {answer}\n")
                safe_update_jsonl('chat_history.jsonl', 'name', patient.disease_name, 'chat_history', {"Q": question, "A": answer})

            elif "[!Diag!]" in action:
                diagnosis = action.split("[!Diag!]")[1].strip().strip("()")
                chat_history += f"Doctor: {diagnosis}\n"
                pos = assistant.get_positive_findings_number() + patient.get_positive_symptoms_number()
                neg = assistant.get_negative_findings_number() + patient.get_negative_symptoms_number()
                print(f"Positive Findings: {pos}")
                print(f"Negative Findings: {neg}")
                safe_append_jsonl(f'result_{doctor.model}.jsonl', {
                    "name": patient.disease_name,
                    "final_answer": diagnosis,
                    "turns": 15 - turns,
                    "positive_findings": pos,
                    "negative_findings": neg,
                    "chat_history": chat_history,
                })

        except Exception as e:
            print(f"An error occurred: {e}")
            continue

def main():
    doctor_model = args.model
    patient_model = 'gpt-5'

    with open('../source/data.jsonl', 'r', encoding='utf-8') as f:
        disease_data = [json.loads(line) for line in f]

    result_file = f'result_{doctor_model}.jsonl'
    if os.path.exists(result_file):
        with open(result_file, 'r', encoding='utf-8') as f:
            existing_results = [json.loads(line) for line in f]
        done = {r['name'] for r in existing_results}
        disease_data = [d for d in disease_data if d['name'] not in done]
    else:
        existing_results = []

    if not disease_data:
        print("All diseases have been processed. Exiting.")
        return

    for item in tqdm(disease_data):
        def load_history(filename, field):
            if not os.path.exists(filename): return []
            with open(filename, 'r', encoding='utf-8') as f:
                for line in f:
                    obj = json.loads(line)
                    if obj['name'] == item['name']:
                        return obj.get(field, [])
            return []

        chat_history = load_history('chat_history.jsonl', 'chat_history')
        examination_history = load_history('examination_history.jsonl', 'examination_history')
        checkup_history = load_history('checkup_history.jsonl', 'checkup_history')

        patient = Patient(model=patient_model, disease_description=item['desc'], disease_name=item['name'], chat_list=chat_history)
        doctor = Doctor(model=doctor_model)
        assistant = Assistant(model=patient_model, disease_description=item['desc'], examination_list=examination_history, checkup_list=checkup_history)
        interact(patient, doctor, assistant)

if __name__ == "__main__":
    main()
