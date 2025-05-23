import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def generate_answer(model, tokenizer, question: str, max_new_tokens=50):
    prompt = f"Вопрос: {question}\nПожалуйста, дай развернутый и точный ответ.\nОтвет:"
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    input_len = input_ids.shape[-1]

    # Основное предсказание (beam search, точное)
    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=input_len + max_new_tokens,
        num_beams=3,
        early_stopping=True,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        do_sample=False,
    )

    raw = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = raw[len(prompt):].split('\n')[0].strip()

    # Костыль: если модель просто повторила вопрос — пробуем ещё раз, но с сэмплированием
    if answer.lower() == question.lower():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=input_len + max_new_tokens,
            do_sample=True,
            top_k=50,
            top_p=0.9,
            temperature=0.8,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        raw = tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = raw[len(prompt):].split('\n')[0].strip()

    return answer




def interactive_mode():
    model_path = "tests/model"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    model.eval()

    print("Бот готов! Напишите 'выход' для завершения.")
    while True:
        question = input("Вы: ")
        if question.strip().lower() in {"выход", "exit", "quit"}:
            print("До свидания!")
            break
        answer = generate_answer(model, tokenizer, question)
        print(f"Бот: {answer}\n")


if __name__ == "__main__":
    interactive_mode()