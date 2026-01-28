from llm.request import send_request

class Doctor:
    def __init__(self, model):
        self.model = model
        self.has_final_answer = False  # Tracks if a final diagnosis has been provided
        self.final_answer = None  # Stores the final diagnosis


    def get_final_answer(self):
        """
        Returns the final diagnosis if available.
        """
        return self.final_answer
    
    def check_finish(self):
        """
        Check if the doctor has provided a final answer.
        """
        return self.has_final_answer

    def provide_diagnosis(self, chat_history):
        """
        Provide a diagnosis based on the patient's description.
        """
        prompt = f"""You are a professional physician. Based on the patient’s consultation records, you need to make a judgment. Your goal is to simulate a routine outpatient visit, rule out similar diseases, and diagnose the disease the patient has. You have the following actions available:

Ask the patient for information, format: [!Ask!](your question, only one question at a time)

Perform a physical examination, format: [!Ask!](the examination you need, only one item at a time)

Conduct auxiliary tests, format: [!Test!](the test you need, only one item at a time)

Provide a diagnosis, format: [!Test!](your diagnosis, must be a single disease name)

You may choose only one action at a time. Once you provide a diagnosis, this will be your final answer, and you will no longer ask the patient for information or perform further tests.

You need to answer after careful reasoning, strictly following the format below:

Thought: (your reasoning process)
Action: [!Ask!](your question) or [!Test!](the test you need) or [!Diag!](your diagnosis)

The consultation records are as follows: {chat_history}

Based on this information, provide your thought process and action. You may only provide one question or one test at a time.
"""

        message = [
            {"role": "system", "content": "You are a doctor."},
            {"role": "user", "content": prompt},
        ]
        response = send_request(message=message, model=self.model)
        
        # Check if the response contains a final diagnosis
        if "[!Diag!]" in response:
            self.has_final_answer = True
        
        return response