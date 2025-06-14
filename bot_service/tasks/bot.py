# tasks/bot.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import re
import os
from typing import Dict, Any, Optional
from celery import Task
from .celery_app import celery_app
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelSingleton:
    """Синглтон для хранения загруженной модели в памяти воркера"""
    _instance = None
    _model = None
    _tokenizer = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelSingleton, cls).__new__(cls)
        return cls._instance

    def initialize_model(self):
        """Инициализация модели (вызывается один раз при старте воркера)"""
        if self._initialized:
            return
        
        try:
            logger.info("Инициализация модели...")
            
            # Базовая модель
            base_model_name = "ai-forever/rugpt3medium_based_on_gpt2"
            
            # Поиск адаптера
            possible_paths = [
                "models/faq_bot/final_model",
                "models/faq_bot",
                "checkpoints/checkpoint_best",
                "checkpoints/checkpoint_last"
            ]
            
            peft_model_path = None
            for path in possible_paths:
                if os.path.exists(os.path.join(path, "adapter_config.json")):
                    peft_model_path = path
                    logger.info(f"Найден адаптер в: {path}")
                    break
            
            # Загрузка токенизатора
            if peft_model_path and os.path.exists(os.path.join(peft_model_path, "tokenizer.json")):
                self._tokenizer = AutoTokenizer.from_pretrained(peft_model_path)
                logger.info("Токенизатор загружен из адаптера")
            else:
                self._tokenizer = AutoTokenizer.from_pretrained(base_model_name)
                logger.info("Токенизатор загружен из базовой модели")
            
            # Устанавливаем pad_token
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token
            
            # Загрузка базовой модели
            logger.info("Загружаем базовую модель...")
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            
            # Загрузка PEFT адаптера
            if peft_model_path:
                try:
                    self._model = PeftModel.from_pretrained(base_model, peft_model_path)
                    logger.info("PEFT адаптер успешно загружен!")
                except Exception as e:
                    logger.error(f"Ошибка загрузки адаптера: {e}")
                    self._model = base_model
            else:
                self._model = base_model
            
            self._model.eval()
            self._initialized = True
            logger.info("Модель успешно инициализирована!")
            
        except Exception as e:
            logger.error(f"Ошибка инициализации модели: {e}")
            raise

    @property
    def model(self):
        if not self._initialized:
            self.initialize_model()
        return self._model

    @property
    def tokenizer(self):
        if not self._initialized:
            self.initialize_model()
        return self._tokenizer

# Глобальный экземпляр модели
model_instance = ModelSingleton()

class ModelTask(Task):
    """Базовый класс для задач с моделью"""
    
    def on_failure(self, exc, task_id, args, kwargs, einfo):
        logger.error(f"Задача {task_id} завершилась с ошибкой: {exc}")
    
    def on_success(self, retval, task_id, args, kwargs):
        logger.info(f"Задача {task_id} успешно выполнена")

@celery_app.task(bind=True, base=ModelTask, name='bot_service.tasks.bot.process_question')
def process_question(self, question: str, user_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Обработка вопроса пользователя
    
    Args:
        question: Вопрос пользователя
        user_id: ID пользователя (опционально)
    
    Returns:
        Dict с ответом и метаданными
    """
    try:
        logger.info(f"Обработка вопроса от пользователя {user_id}: {question}")
        
        # Обновляем статус задачи
        self.update_state(state='PROCESSING', meta={'message': 'Анализирую вопрос...'})
        
        # Проверяем, связан ли вопрос с УрФУ
        urfu_keywords = [
            'урфу', 'уральский', 'федеральный', 'университет', 'студент', 'учеба', 
            'поступление', 'экзамен', 'деканат', 'факультет', 'кафедра', 'общежитие',
            'стипендия', 'сессия', 'диплом', 'справка', 'расписание', 'пропуск',
            'сдача', 'пересдача', 'зачет', 'лекция', 'семинар', 'практика'
        ]
        
        question_lower = question.lower()
        is_urfu_related = any(keyword in question_lower for keyword in urfu_keywords)
        
        if not is_urfu_related:
            return {
                'answer': 'Я отвечаю только на вопросы, связанные с Уральским федеральным университетом (УрФУ). Пожалуйста, задайте вопрос о УрФУ, учебе или студенческой жизни.',
                'confidence': 0.0,
                'is_urfu_related': False,
                'processing_time': 0.1,
                'user_id': user_id
            }
        
        # Обновляем статус
        self.update_state(state='PROCESSING', meta={'message': 'Генерирую ответ...'})
        
        # Создаем промпт
        prompt = f"""Ты - помощник по вопросам Уральского федерального университета (УрФУ).
Отвечай только на вопросы связанные с УрФУ, учебой, поступлением, студенческой жизнью.
Давай краткие и точные ответы.

Вопрос: {question}
Ответ:"""
        
        # Токенизация
        inputs = model_instance.tokenizer(
            prompt, 
            return_tensors="pt", 
            padding=True, 
            truncation=True,
            max_length=512
        )
        
        # Перемещаем на устройство модели
        inputs = {k: v.to(model_instance.model.device) for k, v in inputs.items()}
        
        # Генерация ответа
        with torch.no_grad():
            outputs = model_instance.model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=True,
                temperature=0.3,
                top_p=0.9,
                top_k=50,
                repetition_penalty=1.2,
                no_repeat_ngram_size=2,
                eos_token_id=model_instance.tokenizer.eos_token_id,
                pad_token_id=model_instance.tokenizer.pad_token_id,
                use_cache=True,
            )
        
        # Декодирование результата
        full_text = model_instance.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Извлечение ответа
        if "Ответ:" in full_text:
            answer = full_text.split("Ответ:", 1)[1].strip()
        else:
            answer = full_text[len(prompt):].strip()
        
        # Постобработка ответа
        answer = clean_answer(answer)
        
        # Проверка качества ответа
        if len(answer) < 3 or answer.count('<<') > 0 or answer.count('?') > 3:
            answer = "Извините, не смог найти подходящий ответ на ваш вопрос о УрФУ. Попробуйте переформулировать вопрос."
            confidence = 0.2
        else:
            confidence = calculate_confidence(answer, question)
        
        result = {
            'answer': answer,
            'confidence': confidence,
            'is_urfu_related': True,
            'user_id': user_id,
            'question': question
        }
        
        logger.info(f"Ответ сгенерирован для пользователя {user_id}")
        return result
        
    except Exception as e:
        logger.error(f"Ошибка при обработке вопроса: {e}")
        return {
            'answer': 'Произошла ошибка при обработке вопроса. Попробуйте еще раз.',
            'confidence': 0.0,
            'is_urfu_related': True,
            'error': str(e),
            'user_id': user_id
        }

def clean_answer(answer: str) -> str:
    """Очистка ответа от артефактов"""
    # Убираем повторяющиеся части
    answer = answer.split("Вопрос:")[0].strip()
    answer = answer.split("?")[0].strip()
    
    # Берем только первое предложение, если ответ слишком длинный
    sentences = answer.split('.')
    if len(sentences) > 1 and len(answer) > 150:
        answer = sentences[0].strip() + '.'
    
    # Очистка от артефактов
    answer = re.sub(r'<[^>]+>', '', answer)  # XML-теги
    answer = re.sub(r'\s+', ' ', answer)     # Нормализация пробелов
    answer = re.sub(r'Ты знаешь.*?', '', answer).strip()
    answer = re.sub(r'Как правильно.*?', '', answer).strip()
    
    return answer.strip()

def calculate_confidence(answer: str, question: str) -> float:
    """Вычисление уверенности в ответе"""
    confidence = 0.8  # Базовая уверенность
    
    # Снижаем уверенность за короткие ответы
    if len(answer) < 10:
        confidence -= 0.3
    
    # Снижаем за слишком общие ответы
    generic_phrases = ['деканат', 'приемная комиссия', 'обратиться']
    if any(phrase in answer.lower() for phrase in generic_phrases):
        confidence -= 0.1
    
    # Повышаем за конкретную информацию
    specific_info = ['справка', 'документ', 'процедура', 'сроки']
    if any(info in answer.lower() for info in specific_info):
        confidence += 0.1
    
    return max(0.1, min(1.0, confidence))

@celery_app.task(name='bot_service.tasks.bot.health_check')
def health_check() -> Dict[str, Any]:
    """Проверка состояния бота"""
    try:
        # Проверяем, что модель инициализирована
        model_ready = model_instance._initialized
        
        return {
            'status': 'healthy',
            'model_ready': model_ready,
            'timestamp': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'
        }
    except Exception as e:
        return {
            'status': 'unhealthy',
            'error': str(e),
            'model_ready': False
        }

@celery_app.task(bind=True, name='bot_service.tasks.bot.batch_process')
def batch_process_questions(self, questions: list, user_id: Optional[str] = None) -> Dict[str, Any]:
    """Пакетная обработка вопросов"""
    results = []
    total_questions = len(questions)
    
    for i, question in enumerate(questions):
        self.update_state(
            state='PROCESSING', 
            meta={
                'current': i + 1, 
                'total': total_questions, 
                'message': f'Обрабатываю вопрос {i + 1} из {total_questions}'
            }
        )
        
        result = process_question.apply_async(args=[question, user_id]).get()
        results.append(result)
    
    return {
        'results': results,
        'total_processed': total_questions,
        'user_id': user_id
    }