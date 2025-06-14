import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import re
import os

def main():
    # Сначала загружаем базовую модель и токенизатор
    base_model_name = "ai-forever/rugpt3medium_based_on_gpt2"
    
    # Ищем правильный путь к адаптеру
    possible_paths = [
        "models/faq_bot",
        "models/faq_bot/final_model", 
        "checkpoints/checkpoint_best",
        "checkpoints/checkpoint_last"
    ]
    
    peft_model_path = None
    for path in possible_paths:
        if os.path.exists(os.path.join(path, "adapter_config.json")):
            peft_model_path = path
            print(f"Найден адаптер в: {path}")
            break
    
    if peft_model_path is None:
        print("Адаптер не найден! Будет использована только базовая модель.")
    
    print("Загружаем токенизатор базовой модели...")
    # Попробуем загрузить токенизатор из адаптера, если есть, иначе из базовой модели
    if peft_model_path and os.path.exists(os.path.join(peft_model_path, "tokenizer.json")):
        tokenizer = AutoTokenizer.from_pretrained(peft_model_path)
        print("Токенизатор загружен из адаптера")
    else:
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        print("Токенизатор загружен из базовой модели")
    
    # Устанавливаем pad_token если его нет
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("Загружаем базовую модель...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    print("Загружаем PEFT адаптер...")
    if peft_model_path:
        try:
            model = PeftModel.from_pretrained(base_model, peft_model_path)
            print("Адаптер успешно загружен!")
        except Exception as e:
            print(f"Ошибка загрузки адаптера: {e}")
            print("Используем базовую модель без адаптера...")
            model = base_model
    else:
        print("Используем базовую модель без адаптера...")
        model = base_model
    
    model.eval()

    system_prompt = "Ты — UrFU FAQ‑бот. Отвечаешь только на вопросы про УрФУ и студенческую жизнь."
    
    # Определяем стоп-токены
    stop_tokens = ["<<", "<|", "User", "System", "Assistant"]
    
    print("Модель загружена. Можете задавать вопросы!")
    print("Для выхода введите 'exit' или 'quit'")
    
    while True:
        question = input("\nВведите вопрос: ").strip()
        if question.lower() in ("exit", "quit", "выход"):
            break
        
            # Проверяем, связан ли вопрос с УрФУ
            urfu_keywords = ['урфу', 'уральский', 'федеральный', 'университет', 'студент', 'учеба', 
                           'поступление', 'экзамен', 'деканат', 'факультет', 'кафедра', 'общежитие',
                           'стипендия', 'сессия', 'диплом', 'справка', 'расписание']
            
            question_lower = question.lower()
            is_urfu_related = any(keyword in question_lower for keyword in urfu_keywords)
            
            # Если вопрос не связан с УрФУ, даем соответствующий ответ
            if not is_urfu_related:
                print("Ответ: Я отвечаю только на вопросы, связанные с Уральским федеральным университетом (УрФУ). Пожалуйста, задайте вопрос о УрФУ, учебе или студенческой жизни.")
                continue

        # Используем более специфичный промпт для УрФУ
        prompt = f"""Ты - помощник по вопросам Уральского федерального университета (УрФУ).
Отвечай только на вопросы связанные с УрФУ, учебой, поступлением, студенческой жизнью.
Если вопрос не связан с УрФУ, вежливо скажи, что отвечаешь только на вопросы про УрФУ.

Вопрос: {question}
Ответ:"""
        
        # Токенизируем
        inputs = tokenizer(
            prompt, 
            return_tensors="pt", 
            padding=True, 
            truncation=True,
            max_length=512
        )
        
        # Перемещаем на устройство модели
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        try:
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=100,  # Увеличиваем для более полных ответов
                    do_sample=True,
                    temperature=0.3,    # Еще больше снижаем для точности
                    top_p=0.9,          # Увеличиваем для разнообразия
                    top_k=50,           # Увеличиваем для лучшего выбора
                    repetition_penalty=1.2,  # Умеренный штраф
                    no_repeat_ngram_size=2,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                    use_cache=True,
                    # Убираем early_stopping - не поддерживается
                )

            # Декодируем результат
            full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Извлекаем ответ после промпта
            if "Ответ:" in full_text:
                answer = full_text.split("Ответ:", 1)[1].strip()
            else:
                answer = full_text[len(prompt):].strip()
            
            # Очищаем ответ - берем только до первого вопроса или странных символов
            answer = answer.split("Вопрос:")[0].strip()
            answer = answer.split("?")[0].strip()  # Останавливаемся на первом вопросе
            if "?" in answer and len(answer) > 100:  # Если есть вопрос и текст длинный
                answer = answer.split("?")[0].strip()
            
            # Берем только первое предложение, если ответ слишком длинный
            sentences = answer.split('.')
            if len(sentences) > 1 and len(answer) > 100:
                answer = sentences[0].strip() + '.'
            
            # Очищаем ответ от стоп-токенов
            for stop_token in stop_tokens:
                if stop_token in answer:
                    answer = answer.split(stop_token)[0].strip()
            
            # Дополнительная очистка от артефактов
            answer = re.sub(r'<[^>]+>', '', answer)  # Удаляем XML-теги
            answer = re.sub(r'\s+', ' ', answer)     # Нормализуем пробелы
            answer = answer.strip()
            
            # Убираем повторяющиеся фразы и вопросы
            answer = re.sub(r'Ты знаешь.*?', '', answer).strip()
            answer = re.sub(r'Как правильно.*?', '', answer).strip()
            
            # Проверяем качество ответа - делаем проверку менее строгой
            bad_indicators = ['team', 'assistance', 'curriculum', 'partner']
            
            # Проверяем длину и качество
            if (len(answer) < 3 or 
                answer.count('<<') > 0 or
                answer.count('?') > 3):  # Много вопросов подряд
                print("Ответ: Извините, не смог найти подходящий ответ на ваш вопрос о УрФУ. Попробуйте переформулировать вопрос.")
            else:
                print(f"Ответ: {answer}")
                
        except Exception as e:
            print(f"Ошибка при генерации: {e}")
            print("Ответ: Произошла ошибка при обработке вопроса.")

def check_model_files():
    """Проверяем наличие необходимых файлов модели"""
    peft_path = "models/faq_bot"
    required_files = ["adapter_config.json", "adapter_model.safetensors"]
    
    print("Проверка файлов модели:")
    for file in required_files:
        file_path = os.path.join(peft_path, file)
        if os.path.exists(file_path):
            print(f"✓ {file} найден в {peft_path}")
        else:
            print(f"✗ {file} НЕ найден в {peft_path}")
    
    print(f"\nСодержимое папки {peft_path}:")
    if os.path.exists(peft_path):
        for item in os.listdir(peft_path):
            item_path = os.path.join(peft_path, item)
            if os.path.isdir(item_path):
                print(f"  📁 {item}/")
                # Проверяем содержимое подпапок
                try:
                    for subitem in os.listdir(item_path):
                        print(f"    - {subitem}")
                except:
                    pass
            else:
                print(f"  📄 {item}")
    else:
        print("  Папка не существует!")

if __name__ == "__main__":
    check_model_files()
    print("\n" + "="*50 + "\n")
    main()