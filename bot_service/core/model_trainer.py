import json
import logging
import torch
from torch.optim import AdamW
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, 
    get_linear_schedule_with_warmup,
    TrainingArguments, Trainer
)
from peft import LoraConfig, get_peft_model, TaskType
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import time
import pickle
from tqdm import tqdm
import argparse
import yaml
import os
import gc

from faq_parser import FAQDatasetParser, TrainingExample

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TrainingConfig:
    """Конфигурация обучения"""
    # Модель
    model_name: str = "ai-forever/rugpt3medium_based_on_gpt2"
    max_length: int = 512
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    
    # Обучение  
    learning_rate: float = 2e-4
    num_epochs: int = 3
    batch_size: int = 4  # Для RTX 2060
    gradient_accumulation_steps: int = 4
    warmup_steps: int = 100
    weight_decay: float = 0.01
    
    # Валидация и сохранение
    eval_steps: int = 500
    save_steps: int = 1000
    early_stopping_patience: int = 3
    
    # Пути
    data_dir: Path = Path("data/faq")
    output_dir: Path = Path("models/faq_bot")
    checkpoint_dir: Path = Path("checkpoints")
    
    # Непрерывное обучение
    continuous_training: bool = False
    scan_interval: int = 3600  # секунды
    
    # Off-topic фильтрация
    off_topic_threshold: float = 0.7
    off_topic_model_path: Path = Path("models/off_topic_classifier.pkl")

class FAQDataset(Dataset):
    """Dataset для FAQ обучения"""
    
    def __init__(self, examples: List[TrainingExample], tokenizer, max_length: int = 512):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        self.system_prompt = "<<System>> Ты — UrFU FAQ-бот. Отвечаешь только на вопросы про УрФУ и студенческую жизнь. <<User>> {question} <<Assistant>> {answer}"
        
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        
        text = self.system_prompt.format(
            question=example.question,
            answer=example.answer
        )
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": encoding["input_ids"].flatten().clone()  # Для causal LM
        }

class OffTopicClassifier:
    """Классификатор для фильтрации off-topic вопросов"""
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
        self.classifier = LogisticRegression()
        self.is_trained = False
    
    def prepare_training_data(self) -> Tuple[List[str], List[int]]:
        """Подготовка данных для обучения классификатора"""
        # Примеры по теме (1) и не по теме (0)
        on_topic_examples = [
            "Как поступить в УрФУ?",
            "Какие факультеты есть в университете?", 
            "Где находится общежитие?",
            "Как получить справку в деканате?",
            "Расписание занятий на завтра",
            "Стипендия когда будет?",
            "Где столовая в главном корпусе?",
            "Как записаться на пересдачу?",
            "Контакты приемной комиссии",
            "График работы библиотеки"
        ]
        
        off_topic_examples = [
            "Как приготовить борщ?",
            "Погода в Москве сегодня",
            "Курс доллара к рублю",
            "Последние новости спорта",
            "Рецепт пиццы маргарита",
            "Как починить компьютер?",
            "Лучшие фильмы 2024 года",
            "Как похудеть быстро?",
            "Цены на недвижимость",
            "Что такое криптовалюта?"
        ]
        
        texts = on_topic_examples + off_topic_examples
        labels = [1] * len(on_topic_examples) + [0] * len(off_topic_examples)
        
        return texts, labels
    
    def train(self):
        """Обучение классификатора"""
        texts, labels = self.prepare_training_data()
        
        X = self.vectorizer.fit_transform(texts)
        self.classifier.fit(X, labels)
        self.is_trained = True
        
        logger.info("Off-topic классификатор обучен")
    
    def predict(self, text: str) -> float:
        """Предсказание вероятности, что вопрос по теме"""
        if not self.is_trained:
            return 1.0  # Если не обучен, пропускаем все
        
        X = self.vectorizer.transform([text])
        proba = self.classifier.predict_proba(X)[0][1]  # Вероятность класса 1 (по теме)
        return proba
    
    def save(self, path: Path):
        """Сохранение модели"""
        with open(path, 'wb') as f:
            pickle.dump({
                'vectorizer': self.vectorizer,
                'classifier': self.classifier,
                'is_trained': self.is_trained
            }, f)
    
    def load(self, path: Path):
        """Загрузка модели"""
        if path.exists():
            with open(path, 'rb') as f:
                data = pickle.load(f)
                self.vectorizer = data['vectorizer']
                self.classifier = data['classifier'] 
                self.is_trained = data['is_trained']

class FAQModelTrainer:
    """Основной класс для обучения FAQ модели"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Используется устройство: {self.device}")
        
        # Инициализация компонентов
        self.tokenizer = None
        self.model = None
        self.off_topic_classifier = OffTopicClassifier()
        
        # Статистика обучения
        self.training_stats = {
            'total_examples': 0,
            'epochs_completed': 0,
            'best_eval_loss': float('inf'),
            'early_stopping_counter': 0
        }
        
        # Создание директорий
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        self.config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def load_model_and_tokenizer(self):
        """Загрузка модели и токенизатора"""
        logger.info(f"Загрузка модели: {self.config.model_name}")
        
        # Токенизатор
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Модель (обязательно float32 для RTX 2060)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=torch.float32,  # Без fp16!
            device_map="auto",
            low_cpu_mem_usage=True
        )
        
        # Gradient checkpointing для экономии памяти
        self.model.gradient_checkpointing_enable()
        
        # LoRA настройка
        if self.config.use_lora:
            logger.info("Применение LoRA конфигурации")
            lora_config = LoraConfig(
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                target_modules=["c_attn", "c_proj"],  # Для GPT-2 архитектуры
                lora_dropout=self.config.lora_dropout,
                bias="none",
                task_type=TaskType.CAUSAL_LM
            )
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()
    
    def ingest_one(self, file_path: Path) -> Optional[FAQDataset]:
        """Загрузка одного файла и создание датасета"""
        logger.info(f"Обработка файла: {file_path}")
        
        parser = FAQDatasetParser()
        examples = parser.load_file(file_path)
        
        if not examples:
            logger.warning(f"Нет примеров в файле {file_path}")
            return None
        
        # Фильтрация
        parser.examples = examples
        parser.filter_examples(
            min_question_length=10,
            min_answer_length=15,
            exclude_negative=True
        )
        
        if not parser.examples:
            logger.warning(f"После фильтрации не осталось примеров в {file_path}")
            return None
        
        logger.info(f"Загружено {len(parser.examples)} примеров из {file_path}")
        return FAQDataset(parser.examples, self.tokenizer, self.config.max_length)
    
    def train_one_file(self, file_path: Path):
        """Обучение на одном файле"""
        dataset = self.ingest_one(file_path)
        if not dataset:
            return
        
        # Разделение на train/val
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )
        
        # DataLoaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=2
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=2
        ) if val_size > 0 else None
        
        # Обучение
        self._train_on_dataloader(train_loader, val_loader)
        
        # Очистка памяти
        del dataset, train_dataset, val_dataset, train_loader, val_loader
        gc.collect()
        torch.cuda.empty_cache()
    
    def _train_on_dataloader(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None):
        """Основной цикл обучения"""
        # Оптимизатор
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        # Scheduler
        total_steps = len(train_loader) * self.config.num_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=total_steps
        )
        
        self.model.train()
        global_step = 0
        
        for epoch in range(self.config.num_epochs):
            logger.info(f"Эпоха {epoch + 1}/{self.config.num_epochs}")
            
            epoch_loss = 0
            progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch + 1}")
            
            for step, batch in enumerate(progress_bar):
                # Перенос на GPU
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(**batch)
                loss = outputs.loss
                
                # Gradient accumulation
                loss = loss / self.config.gradient_accumulation_steps
                loss.backward()
                
                epoch_loss += loss.item()
                
                if (step + 1) % self.config.gradient_accumulation_steps == 0:
                    # Clipping градиентов
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1
                
                # Обновление прогресс-бара
                progress_bar.set_postfix({'loss': loss.item() * self.config.gradient_accumulation_steps})
                
                # Валидация
                if val_loader and global_step % self.config.eval_steps == 0:
                    eval_loss = self._evaluate(val_loader)
                    logger.info(f"Step {global_step}, Eval Loss: {eval_loss:.4f}")
                    
                    # Early stopping
                    if eval_loss < self.training_stats['best_eval_loss']:
                        self.training_stats['best_eval_loss'] = eval_loss
                        self.training_stats['early_stopping_counter'] = 0
                        self._save_checkpoint('best')
                    else:
                        self.training_stats['early_stopping_counter'] += 1
                    
                    if self.training_stats['early_stopping_counter'] >= self.config.early_stopping_patience:
                        logger.info("Early stopping triggered")
                        return
                
                # Сохранение чекпоинта
                if global_step % self.config.save_steps == 0:
                    self._save_checkpoint('last')
            
            avg_epoch_loss = epoch_loss / len(train_loader)
            logger.info(f"Средняя потеря за эпоху: {avg_epoch_loss:.4f}")
            self.training_stats['epochs_completed'] += 1
    
    def _evaluate(self, val_loader: DataLoader) -> float:
        """Валидация модели"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                total_loss += outputs.loss.item()
        
        self.model.train()
        return total_loss / len(val_loader)
    
    def _save_checkpoint(self, checkpoint_type: str = 'last'):
        """Сохранение чекпоинта"""
        checkpoint_path = self.config.checkpoint_dir / f"checkpoint_{checkpoint_type}"
        
        if self.config.use_lora:
            # Сохранение только LoRA весов
            self.model.save_pretrained(checkpoint_path)
        else:
            # Сохранение полной модели
            self.model.save_pretrained(checkpoint_path)
        
        self.tokenizer.save_pretrained(checkpoint_path)
        
        # Сохранение статистики
        stats_path = checkpoint_path / "training_stats.json"
        with open(stats_path, 'w') as f:
            json.dump(self.training_stats, f, indent=2)
        
        logger.info(f"Чекпоинт сохранен: {checkpoint_path}")
    
    def train_all_files(self):
        """Обучение на всех файлах поочередно"""
        json_files = list(self.config.data_dir.glob("*.json"))
        
        if not json_files:
            logger.error(f"JSON файлы не найдены в {self.config.data_dir}")
            return
        
        logger.info(f"Найдено {len(json_files)} файлов для обучения")
        
        # Загрузка модели
        if not self.model:
            self.load_model_and_tokenizer()
        
        # Обучение off-topic классификатора
        self.off_topic_classifier.train()
        self.off_topic_classifier.save(self.config.off_topic_model_path)
        
        # Обучение на каждом файле
        for json_file in json_files:
            logger.info(f"Обучение на файле: {json_file}")

            # **сбрасываем ранний стоп и лучшую метрику** для нового файла
            self.training_stats['early_stopping_counter'] = 0
            self.training_stats['best_eval_loss'] = float('inf')

            self.train_one_file(json_file)
        
        # Финальное сохранение
        final_path = self.config.output_dir / "final_model"
        if self.config.use_lora:
            self.model.save_pretrained(final_path)
        else:
            self.model.save_pretrained(final_path)
        self.tokenizer.save_pretrained(final_path)
        
        logger.info(f"Обучение завершено. Модель сохранена в {final_path}")
    
    def continuous_training_loop(self):
        """Цикл непрерывного обучения"""
        if not self.config.continuous_training:
            return
        
        logger.info("Запуск режима непрерывного обучения")
        processed_files = set()
        
        while True:
            # Сканирование новых файлов
            json_files = set(self.config.data_dir.glob("*.json"))
            new_files = json_files - processed_files
            
            if new_files:
                logger.info(f"Найдено {len(new_files)} новых файлов")
                for new_file in new_files:
                    self.train_one_file(new_file)
                    processed_files.add(new_file)
            
            # Ожидание следующего сканирования
            time.sleep(self.config.scan_interval)
    
    def generate_response(self, question: str, max_length: int = 200) -> str:
        """Генерация ответа на вопрос"""
        # Проверка на off-topic
        if self.off_topic_classifier.is_trained:
            topic_score = self.off_topic_classifier.predict(question)
            if topic_score < self.config.off_topic_threshold:
                return "Извините, я отвечаю только на вопросы про УрФУ и студенческую жизнь."
        
        # Формирование промпта
        prompt = f"<<System>> Ты — UrFU FAQ-бот. Отвечаешь только на вопросы про УрФУ и студенческую жизнь. <<User>> {question} <<Assistant>>"
        
        # Токенизация
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Генерация
        self.model.eval()
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=len(inputs['input_ids'][0]) + max_length,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Декодирование ответа
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Извлечение только ответа модели
        if "<<Assistant>>" in response:
            response = response.split("<<Assistant>>")[-1].strip()
        
        return response

def load_config(config_path: str) -> TrainingConfig:
    """Загрузка конфигурации из YAML файла"""
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        
        # Преобразование путей в Path объекты
        for key in ['data_dir', 'output_dir', 'checkpoint_dir', 'off_topic_model_path']:
            if key in config_dict:
                config_dict[key] = Path(config_dict[key])
        
        return TrainingConfig(**config_dict)
    else:
        return TrainingConfig()

def main():
    """CLI интерфейс"""
    parser = argparse.ArgumentParser(description="FAQ Model Trainer")
    parser.add_argument('--config', default='config.yaml', help='Путь к конфигу')
    parser.add_argument('--dataset', type=str, help='Путь к конкретному датасету')
    parser.add_argument('--all', action='store_true', help='Обучить на всех файлах')
    parser.add_argument('--continuous', action='store_true', help='Режим непрерывного обучения')
    parser.add_argument('--test', type=str, help='Тестовый вопрос для модели')
    
    args = parser.parse_args()
    
    # Загрузка конфигурации
    config = load_config(args.config)
    
    # Инициализация тренера
    trainer = FAQModelTrainer(config)
    
    if args.test:
        # Тестирование модели
        trainer.load_model_and_tokenizer()
        checkpoint_path = config.output_dir / "final_model"
        if checkpoint_path.exists():
            if config.use_lora:
                from peft import PeftModel
                trainer.model = PeftModel.from_pretrained(trainer.model, checkpoint_path)
        
        response = trainer.generate_response(args.test)
        print(f"Вопрос: {args.test}")
        print(f"Ответ: {response}")
        
    elif args.dataset:
        # Обучение на одном файле с продолжением обучения
        trainer.load_model_and_tokenizer()

        # Если есть предыдущий LoRA‑чекпоинт — подгружаем его
        last_ckpt = trainer.config.checkpoint_dir / "checkpoint_best"
        if last_ckpt.exists():
            from peft import PeftModel
            trainer.model = PeftModel.from_pretrained(trainer.model, last_ckpt)
            trainer.model.to(trainer.device)
            logger.info(f"Продолжаем обучение с {last_ckpt}")

        trainer.train_one_file(Path(args.dataset))
        
    elif args.all:
        # Обучение на всех файлах
        trainer.train_all_files()
        
        if args.continuous:
            trainer.continuous_training_loop()
    else:
        print("Используйте --dataset <path>, --all или --test <question>")

if __name__ == "__main__":
    main()