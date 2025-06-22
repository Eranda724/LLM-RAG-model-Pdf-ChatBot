from typing import List
from ctransformers import AutoModelForCausalLM

llm = AutoModelForCausalLM.from_pretrained(
    "TheBloke/Llama-2-7b-Chat-GGUF", 
    model_file="llama-2-7b-chat.Q5_K_M.gguf"
)

def get_prompt(instruction: str, history: List[str] = None) -> str:
    system = "You are an AI assistant that gives helpful answers. You answer the question in a short and concise way. For mathematical questions, provide the exact numerical answer."
    prompt = f"<s>[INST] <<SYS>>\n{system}\n<</SYS>>\n\n"
    if history:
        prompt += f"Previous conversation:\n{''.join(history)}\n\n"
    prompt += f"### User:\n{instruction}\n\n### Response:\n"
    return prompt

class GPTChatBot:
    def __init__(self):
        self.history = []
        self.ready = True

    def invoke(self, prompt: str):
        import time
        start = time.time()
        full_prompt = get_prompt(prompt, self.history)
        answer = ""
        try:
            for word in llm(full_prompt, stream=True):
                answer += word
            self.history.append(f"User: {prompt}\nAssistant: {answer}\n")
            end = time.time()
            return answer, round(end - start, 2)
        except Exception as e:
            return f"Error: {str(e)}", 0

    def clear_history(self):
        self.history = []

    def is_ready(self):
        return self.ready

gpt_chatbot = GPTChatBot()