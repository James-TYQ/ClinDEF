from llm.request import send_request
import re

class Assistant:
    def __init__(self, model, disease_description, examination_list, checkup_list):
        self.model = model
        self.disease_description = disease_description
        self.examination_list = examination_list
        self.checkup_list = checkup_list
        self.positive_findings_number = 0
        self.negative_findings_number = 0

    def check_up(self, doctor_checkup):
        prompt = f"""You are an assistant physician. Your task is to simulate physical examination results based on the clinician’s requested examinations, the disease encyclopedia description, and existing information, combined with your own knowledge of the disease.

The patient’s existing examination information is: {self.examination_list}
The patient’s disease description is: {self.disease_description}

The clinician’s requested physical examinations are: {doctor_checkup}

You need to respond to the clinician’s requested examinations in a professional manner. Based on the clinician’s requests and your understanding of the disease, provide reasonable descriptions of the examination results. Respond directly to the clinician’s requested examinations without including any other information. Objectively describe the results, without providing any biased diagnostic information, disease names, or treatment recommendations. For numerical results, only indicate “normal,” “high,” or “low,” without providing specific values.

For examinations unrelated to the disease, respond with “normal.”

The response format must strictly follow the following format:
[!Positive!](your examination result) or [!Negative!](your examination result)

If the clinician requests multiple examinations, answer each one individually. Each examination result must follow the above principles and be output on separate lines. Example:
[!Positive!](Abdominal palpation result: abdomen soft, tender, no rebound tenderness.)
[!Negative!](Cardiac auscultation result: heart sounds clear, regular rhythm, no murmurs.)
"""

        message = [
            {"role": "system", "content": "You are a doctor."},
            {"role": "user", "content": prompt},
        ]

        response = send_request(message=message, model=self.model)
        # Check if the response contains a positive or negative finding
        
        
        positive_matches = re.findall(r"\[!Positive!\]\((.*?)\)", response, re.DOTALL)
        negative_matches = re.findall(r"\[!Negative!\]\((.*?)\)", response, re.DOTALL)
        self.positive_findings_number += len(positive_matches)
        self.negative_findings_number += len(negative_matches)
        # Combine all findings in order as in the response
        findings = []
        # Use regex to preserve order and content
        for match in re.finditer(r"\[(?P<tag>!Positive!|!Negative!)\]\((?P<content>.*?)\)", response, re.DOTALL):
            tag = match.group("tag")
            content = match.group("content").strip()
            findings.append(f"{content}")
        response = "\n".join(findings)


        # positive_count = len(re.findall(r"\[!Positive!\]", response))
        # negative_count = len(re.findall(r"\[!Negative!\]", response))
        # self.positive_findings_number += positive_count
        # self.negative_symptom_number += negative_count


        return response

    def get_positive_findings_number(self):
        return self.positive_findings_number

    def get_negative_findings_number(self):
        return self.negative_findings_number
