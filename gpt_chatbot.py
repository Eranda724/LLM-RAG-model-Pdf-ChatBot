from langchain_community.llms import CTransformers
from langchain_core.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
import time

class GPTChatBot:
    def __init__(self):
        """Initialize the GPT ChatBot with CTransformers model"""
        try:
            self.llm = CTransformers(
                model="zoltanctoth/orca_mini_3B-GGUF", 
                model_file="orca-mini-3b.q4_0.gguf",
                model_type="llama2",
                max_new_tokens=512,  # Increased for better responses
                temperature=0.7,     # Added temperature for more varied responses
                top_p=0.9,          # Added top_p for better response quality
                top_k=40            # Added top_k for response diversity
            )
            
            self.prompt_template = """###System:
You are an AI assistant that gives helpful, accurate, and informative answers.
Your answers should be clear, concise, and well-structured.
Take this conversation history into account when answering the question:
{context}

### User:
{instruction}

### Response:"""

            self.prompt = PromptTemplate(
                template=self.prompt_template, 
                input_variables=["instruction", "context"]
            )
            
            self.history = []  # List of (user, ai) tuples
            self.initialized = True
            
        except Exception as e:
            print(f"Error initializing GPT ChatBot: {str(e)}")
            self.initialized = False
            self.error_message = str(e)

    def invoke(self, instruction):
        """Get response from the chatbot with conversation memory"""
        try:
            if not self.initialized:
                return f"Error: ChatBot not initialized. {getattr(self, 'error_message', 'Unknown error')}"
            
            # Build context string from history
            context = "\n".join([
                f"Human: {q}\nAI: {a}" for q, a in self.history[-5:]  # Keep last 5 exchanges for context
            ])
            
            prompt_input = {"instruction": instruction, "context": context}
            
            # Get response
            start_time = time.time()
            response = self.llm.invoke(self.prompt.format(**prompt_input))
            end_time = time.time()
            
            # Clean up response (remove any system prefixes)
            if response.startswith("### Response:"):
                response = response.replace("### Response:", "").strip()
            
            # Save to history
            self.history.append((instruction, response))
            
            processing_time = round(end_time - start_time, 2)
            return response, processing_time
            
        except Exception as e:
            return f"Error generating response: {str(e)}", 0

    def clear_history(self):
        """Clear conversation history"""
        self.history = []
        return "Conversation history cleared."

    def get_history(self):
        """Get conversation history"""
        return self.history

    def is_ready(self):
        """Check if the chatbot is ready"""
        return self.initialized

# Create a global instance
gpt_chatbot = GPTChatBot() 