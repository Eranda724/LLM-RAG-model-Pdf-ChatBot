#!/usr/bin/env python3
"""
Test script for the GPT ChatBot
Run this to test if the chatbot is working correctly
"""

from gpt_chatbot import GPTChatBot
import time

def test_gpt_chatbot():
    """Test the GPT ChatBot functionality"""
    print("🤖 Testing GPT ChatBot...")
    print("=" * 50)
    
    # Initialize chatbot
    print("1. Initializing chatbot...")
    chatbot = GPTChatBot()
    
    if not chatbot.is_ready():
        print("❌ ChatBot failed to initialize!")
        print(f"Error: {getattr(chatbot, 'error_message', 'Unknown error')}")
        return False
    
    print("✅ ChatBot initialized successfully!")
    print()
    
    # Test basic questions
    test_questions = [
        "What is 2 + 2?",
        "What is the capital of France?",
        "Explain photosynthesis in simple terms"
    ]
    
    for i, question in enumerate(test_questions, 1):
        print(f"{i}. Testing question: {question}")
        print("-" * 30)
        
        try:
            response, processing_time = chatbot.invoke(question)
            print(f"Response: {response}")
            print(f"Processing time: {processing_time} seconds")
            print()
            
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            print()
    
    # Test conversation memory
    print("4. Testing conversation memory...")
    print("-" * 30)
    
    response1, _ = chatbot.invoke("My name is Alice")
    print(f"Response 1: {response1}")
    
    response2, _ = chatbot.invoke("What's my name?")
    print(f"Response 2: {response2}")
    
    # Test clear history
    print("\n5. Testing clear history...")
    print("-" * 30)
    
    result = chatbot.clear_history()
    print(f"Clear result: {result}")
    
    response3, _ = chatbot.invoke("What's my name?")
    print(f"Response after clear: {response3}")
    
    print("\n" + "=" * 50)
    print("✅ GPT ChatBot test completed successfully!")
    return True

if __name__ == "__main__":
    test_gpt_chatbot() 