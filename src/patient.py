from llm.request import send_request
import re

class Patient:
    def __init__(self, model, disease_name, disease_description, chat_list):
        self.model = model
        self.disease_description = disease_description
        self.chat_list = chat_list
        self.positive_symptoms_number = 0
        self.negative_symptom_number = 0
        self.disease_name = disease_name

    def answer_question(self, doctor_question):

        prompt = f"""You are an ordinary patient, and you are certain that you have the following disease: {self.disease_description}.

Please carefully consider this disease description and answer the doctor’s questions based on your symptoms.
The doctor’s question is: {doctor_question}

You must follow these principles when answering:
Your answer should directly respond to the doctor’s question, simulating a real patient as much as possible to assess the doctor’s professional ability.
Only answer the doctor’s current question; do not provide excessive information. Avoid being too technical or obscure. Do not include any unrelated information.
Do not copy the disease description verbatim. Express it in conversational, everyday, and easy-to-understand language.
If the doctor asks multiple questions at once, answer each question individually. Each answer must follow the above principles.
You must answer according to the doctor’s question. If existing information is incomplete, respond appropriately based on the doctor’s question.

The answer format must strictly follow the following format:
[!Positive!](your answer) or [!Negative!](your answer)

First, determine whether the doctor’s question is relevant to the disease and whether the symptom is present in you. If relevant, output [!Positive!] first; if not relevant, output [!Negative!]. Then provide your answer to the doctor. For multiple questions, output each answer on a separate line. Example:
[!Positive!](I feel a bit of chest pain.)
[!Negative!](I do not feel dizzy.)
"""

        message = [
            {"role": "system", "content": "You are a patient."},
            {"role": "user", "content": prompt},
        ]

        response = send_request(message=message, model="gpt-4o")
        # Check if the response contains a positive or negative answer

         
        positive_matches = re.findall(r"\[!Positive!\]\((.*?)\)", response, re.DOTALL)
        negative_matches = re.findall(r"\[!Negative!\]\((.*?)\)", response, re.DOTALL)
        self.positive_symptoms_number += len(positive_matches)
        self.negative_symptom_number += len(negative_matches)
        # Combine all findings in order as in the response
        findings = []
        # Use regex to preserve order and content
        for match in re.finditer(r"\[(?P<tag>!Positive!|!Negative!)\]\((?P<content>.*?)\)", response, re.DOTALL):
            tag = match.group("tag")
            content = match.group("content").strip()
            findings.append(f"{content}")
        response = "\n".join(findings)


        return response

    def get_positive_symptoms_number(self):
        return self.positive_symptoms_number
    
    def get_negative_symptoms_number(self):
        return self.negative_symptom_number


    def generate_initial_question(self):
        prompt = f"""
You are a standardized patient, and you are certain that you have the following disease: {self.disease_description}.
Based on this disease description, carefully consider and simulate the patient’s first description when seeing a doctor, in order to assess the doctor’s diagnostic ability.
Your description should include only 1–2 main symptoms and be as simple as possible. Wait for the doctor’s questions for other symptoms; do not provide excessive details. Do not include the correct disease name in your description. Keep it within 2 sentences.
Avoid using overly professional terms. Try to simulate the way a real patient would speak, as simple and direct as possible, preferably in a single sentence.
"""

        message = [
            {"role": "system", "content": "You are a patient."},
            {"role": "user", "content": prompt},
        ]
        return send_request(message=message, model=self.model)