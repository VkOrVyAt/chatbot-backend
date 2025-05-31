# bot-service/tasks/bot.py
from celery import Celery
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

app = Celery("bot", broker="redis://localhost:6379/0", backend="redis://localhost:6379/0")
model_name = "sberbank-ai/rugpt3small_based_on_gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to("cuda" if torch.cuda.is_available() else "cpu")

@app.task
def process_message(message: str) -> str:
    inputs = tokenizer(message, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_length=50)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)