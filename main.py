import google.generativeai as genai
import os
import gradio as gr
from time import sleep
from transformers import pipeline

genai.configure(api_key="AIzaSyAA179g3BG3vkfOoAo1a2ty6gSjMjPYbQ4")
model = genai.GenerativeModel("gemini-pro")
sentiment_analyzer = pipeline("sentiment-analysis")

def generate_response(user_input):
    try:
        response = model.generate_content(user_input)
        return response.text
    except Exception as e:
        return "I'm sorry, I encountered an error. Can you please try again?"

def generate_empathic_response(user_input):
    sentiment = sentiment_analyzer(user_input)[0]
    sentiment_label = sentiment['label']
    sentiment_score = sentiment['score']
    
    if sentiment_label == 'NEGATIVE' and sentiment_score > 0.85:
        prompt = f"You are an empathetic virtual assistant. A student is feeling distressed: {user_input}. Offer them emotional support and reassure them."
    elif sentiment_label == 'POSITIVE':
        prompt = f"You are a friendly assistant. The student is feeling positive: {user_input}. Continue offering encouragement."
    else:
        prompt = f"You are a supportive virtual assistant providing emotional guidance to students. Here is the input: {user_input}."
    
    response = generate_response(prompt)
    sleep(1.5)  
    return response

def chatbot_interface(user_input):
    print("Bot is typing...")
    response = generate_empathic_response(user_input)
    return response

theme = gr.themes.Soft(
    primary_hue="green",
    secondary_hue="slate",
    neutral_hue="emerald",
)
gr.Interface(
    fn=chatbot_interface,
    inputs=gr.Textbox(
        lines=4, 
        placeholder="What's on your mind? Share your thoughts or feelings...",
        label="Enter your message"
    ),
    outputs=gr.Textbox(label="Bot's Response"),
    title="🌿 Gemini AI Mental Health Support Chatbot",
    description="A virtual assistant providing empathetic emotional support to students. FOR GEN AI EXCHANGE HACKATHON BY GOOGLE !",
    examples=[
        "I'm feeling really stressed about my exams.",
        "I don't know how to deal with all this anxiety.",
        "I feel overwhelmed with school and life."
    ],
    theme=theme,
    # css=".gradio-container {background-color: white}",
    allow_flagging="never",
).launch(share=True)
