import torch
import logging
from pathlib import Path
from typing import List, Dict, Optional
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
import json
import os
from core.faq_parser import FAQParser

logger = logging.getLogger(__name__)


class FAQModelTrainer:
    """Тренер для fine-tuning русской языковой модели на FAQ данных"""
    
    def __init__(
        self, 
        model_name: str = "ai-forever/rugpt3medium_based_on_gpt2",
        output_dir: str = "models/faq_model",
        max_length: int = 512
    ):
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.max_length = max_length
        self.device = self._get_device()
        
        # Создаем папку для модели
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.tokenizer = None
        self.model = None
        
        logger.info(f"Инициализация тренера. Устройство: {self.device}")
    
    def _get_device(self) -> str:
        """Определяет доступное устройство для обучения"""
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            logger.info(f"Найдена GPU: {gpu_name}")
            return "cuda"
        else:
            logger.warning("GPU не найдена, используется CPU")
            return "cpu"
    
    def load_model_and_tokenizer(self) -> None:
        """Загружает модель и токенизатор"""
        logger.info(f"Загрузка модели: {self.model_name}")
        
        try:
            # Загружаем токенизатор
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                padding_side="left",
                trust_remote_code=True
            )
            
            # Добавляем pad_token если его нет
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Загружаем модель без FP16 для избежания проблем с градиентами
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float32,  # Используем FP32 вместо FP16
                low_cpu_mem_usage=True,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True,
                use_safetensors=True
            )
            
            logger.info("Модель и токенизатор загружены успешно")
            
        except Exception as e:
            logger.error(f"Ошибка загрузки модели: {e}")
            raise
    
    def prepare_training_data(self, faq_data: List[Dict[str, str]]) -> Dataset:
        """Подготавливает данные для обучения"""
        logger.info(f"Подготовка {len(faq_data)} записей для обучения")
        
        # Формируем тренировочные тексты в формате: [ВОПРОС] вопрос [ОТВЕТ] ответ
        texts = []
        for item in faq_data:
            text = f"[ВОПРОС] {item['question']} [ОТВЕТ] {item['answer']}"
            texts.append(text)
        
        # Токенизируем тексты
        def tokenize_function(examples):
            # Токенизируем с padding и truncation
            tokenized = self.tokenizer(
                examples["text"],
                truncation=True,
                padding="max_length",  # Важно: добавляем padding
                max_length=self.max_length,
                return_overflowing_tokens=False,
                return_tensors=None  # Возвращаем списки, не тензоры
            )
            
            # Для causal LM labels = input_ids
            tokenized["labels"] = tokenized["input_ids"].copy()
            return tokenized
        
        # Создаем dataset
        dataset = Dataset.from_dict({"text": texts})
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names
        )
        
        logger.info(f"Подготовлено {len(tokenized_dataset)} примеров")
        return tokenized_dataset
    
    def train(
        self,
        faq_data: List[Dict[str, str]],
        num_epochs: int = 3,
        batch_size: int = 2,  # Маленький batch для RTX 2060
        learning_rate: float = 5e-5,
        save_steps: int = 500,
        logging_steps: int = 100
    ) -> None:
        """Обучает модель на FAQ данных"""
        
        if self.model is None or self.tokenizer is None:
            self.load_model_and_tokenizer()
        
        # Подготавливаем данные
        train_dataset = self.prepare_training_data(faq_data)
        
        # Настройки обучения для RTX 2060
        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            overwrite_output_dir=True,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=4,  # Увеличиваем эффективный batch size
            learning_rate=learning_rate,
            weight_decay=0.01,
            logging_steps=logging_steps,
            save_steps=save_steps,
            save_total_limit=2,
            prediction_loss_only=True,
            remove_unused_columns=False,
            dataloader_pin_memory=False,
            # Исправляем проблему с FP16
            fp16=False,  # Отключаем FP16 чтобы избежать ошибки с градиентами
            bf16=False,  # Также отключаем BF16
            max_grad_norm=1.0,  # Добавляем gradient clipping
            dataloader_num_workers=0,
            group_by_length=False,  # Отключаем группировку по длине
            report_to=None,  # Отключаем wandb
            # Дополнительные настройки для стабильности
            warmup_steps=50,
            logging_first_step=True,
        )
        
        # Коллатор данных с правильными настройками
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,  # Causal LM, не masked LM
            pad_to_multiple_of=None,  # Не выравниваем до кратного числа
            return_tensors="pt"
        )
        
        # Создаем тренер
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=data_collator,
            processing_class=self.tokenizer,  # Используем processing_class вместо tokenizer
        )
        
        logger.info("Начинаем обучение...")
        
        try:
            # Обучение
            trainer.train()
            
            # Сохраняем модель
            logger.info("Сохранение обученной модели...")
            trainer.save_model()
            self.tokenizer.save_pretrained(str(self.output_dir))
            
            # Сохраняем конфигурацию обучения
            config = {
                "base_model": self.model_name,
                "num_epochs": num_epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "max_length": self.max_length,
                "training_samples": len(faq_data),
                "device": self.device
            }
            
            with open(self.output_dir / "training_config.json", "w", encoding="utf-8") as f:
                json.dump(config, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Обучение завершено. Модель сохранена в {self.output_dir}")
            
        except Exception as e:
            logger.error(f"Ошибка во время обучения: {e}")
            raise
    
    def load_trained_model(self) -> None:
        """Загружает обученную модель"""
        
        logger.info(f"Загрузка обученной модели из {self.output_dir}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.output_dir))
        self.model = AutoModelForCausalLM.from_pretrained(
            str(self.output_dir),
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            low_cpu_mem_usage=True,
            device_map="auto" if self.device == "cuda" else None,
        )
        
        logger.info("Обученная модель загружена")
    
    def generate_answer(self, question: str, max_new_tokens: int = 150) -> str:
        """Генерирует ответ на вопрос"""
        if self.model is None or self.tokenizer is None:
            raise ValueError("Модель не загружена")
        
        # Формируем промпт
        prompt = f"[ВОПРОС] {question} [ОТВЕТ]"
        
        # Токенизируем
        inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.model.device)
        
        # Генерируем ответ
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                no_repeat_ngram_size=3
            )
        
        # Декодируем ответ
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Извлекаем только новую часть (ответ)
        answer = generated_text[len(prompt):].strip()
        
        return answer


def main():
    """Основная функция для обучения модели"""
    logging.basicConfig(level=logging.INFO)
    
    # Инициализируем парсер и тренер
    parser = FAQParser()
    trainer = FAQModelTrainer()
    
    # Загружаем данные FAQ
    faq_data = parser.parse_all_files()
    
    if not faq_data:
        logger.error("Нет данных для обучения")
        return
    
    logger.info(f"Загружено {len(faq_data)} записей FAQ")
    
    # Обучаем модель
    trainer.train(
        faq_data=faq_data,
        num_epochs=3,
        batch_size=2,  # Для RTX 2060
        learning_rate=5e-5
    )


if __name__ == "__main__":
    main()