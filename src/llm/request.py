import os
import json
from openai import OpenAI



os.environ["OPENAI_API_BASE"] = ""
os.environ["OPENAI_API_KEY"] = ""


os.environ["DASHSCOPE_API_BASE"] = ""
os.environ["DASHSCOPE_API_KEY"] = ""


os.environ["MODELSCOPE_API_BASE"] = ""
os.environ["MODELSCOPE_API_KEY"] = ''

def send_request(message, model):

    try:
        # stream
        if model in ['qwen3-8b']:

            key = os.getenv('DASHSCOPE_API_KEY')
            url = os.getenv('DASHSCOPE_API_BASE')
            client = OpenAI(
                api_key=key, base_url=url, max_retries=5
            )
            completion = client.chat.completions.create(
                model=model,
                messages=message,
                temperature=0.0,
                stream=True,
            )
              

            reasoning_content = ""  
            answer_content = ""   
            is_answering = False


            for chunk in completion:

                if not chunk.choices:
                    print("\nUsage:")
                    print(chunk.usage)
                else:
                    delta = chunk.choices[0].delta
                    if hasattr(delta, 'reasoning_content') and delta.reasoning_content != None:
                        print(delta.reasoning_content, end='', flush=True)
                        reasoning_content += delta.reasoning_content
                    else:
                        if delta.content != "" and is_answering is False:
                            is_answering = True
                        print(delta.content, end='', flush=True)
                        answer_content += delta.content

            return answer_content

        elif model == "manual":
            print("Please enter your response:")
            response = input()
            return response
        
        # local model call
        elif model in ["qwen2.5-7b"]:
            url = "http://localhost:8000/v1"

            client = OpenAI(
                base_url=url,
                max_retries=5
            )

            completion = client.chat.completions.create(
                model=model,
                messages=message,
                temperature=0.0,
            )
            output = completion.choices[0].message.content

            return output

        # Alibaba cloud model
        elif model in ['qwen3-next-80b-a3b-instruct']:
            key = os.getenv('DASHSCOPE_API_KEY')
            url = os.getenv('DASHSCOPE_API_BASE')


            client = OpenAI(
                api_key=key, base_url=url, max_retries=5
            )


            completion = client.chat.completions.create(
                    model= model, 
                    messages=message,
                    temperature=0.0,
                )
            
            # print(completion)
            output=completion.choices[0].message.content

            # print(output)
            return output
        
        # OpenAI model
        else:
            key = os.getenv('OPENAI_API_KEY')
            url = os.getenv('OPENAI_API_BASE')


            client = OpenAI(
                api_key=key, base_url=url, max_retries=5
            )


            completion = client.chat.completions.create(
                    model= model, 
                    messages=message,
                    # temperature=0.0,
                )
            
            # print(completion)
            output=completion.choices[0].message.content

            # print(output)
            return output
        

    except Exception as e:
        print(f"Error calling API: {e}")
        return None

