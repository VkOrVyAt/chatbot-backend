import json
import logging
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
from tqdm import tqdm


logger = logging.getLogger(__name__)

class DatasetFormat(Enum):
    """Типы форматов датасетов"""
    SIMPLE_QA = "simple_qa"           # Формат 1: question/answer
    CATEGORY_INTENTS = "category_intents"  # Формат 2: category/intents/answer
    ADVANCED_QA = "advanced_qa"       # Формат 3: intentions/variants/good_answer

@dataclass
class TrainingExample:
    """Унифицированная структура для обучающего примера"""
    question: str
    answer: str
    category: Optional[str] = None
    confidence: float = 1.0
    metadata: Optional[Dict[str, Any]] = None

class FAQDatasetParser:
    """Идеальный парсер для FAQ датасетов с очисткой текста и надежной обработкой"""

    def __init__(self, data_dir: Path = Path("data/faq"), max_examples_per_file: int = 1000):
        """Инициализация парсера"""
        self.data_dir = Path(data_dir)
        self.examples: List[TrainingExample] = []
        self.max_examples_per_file = max_examples_per_file

    def clean_text(self, text: str) -> str:
        """Очистка текста: нормализация пробелов, удаление спецсимволов"""
        if not isinstance(text, str):
            return ""
        text = re.sub(r'\s+', ' ', text.strip())
        text = re.sub(r'[^\w\s.,!?()-]', '', text)
        return text

    def detect_format(self, data: Any) -> DatasetFormat:
        """Определение формата JSON с поддержкой ключа 'data'"""
        try:
            if isinstance(data, dict) and "data" in data and isinstance(data["data"], list):
                inner_data = data["data"]
            elif isinstance(data, list):
                inner_data = data
            else:
                logger.error(f"Некорректная структура данных: {type(data)}")
                raise ValueError("Некорректный формат данных!")

            if not inner_data:
                logger.error("Данные пусты")
                raise ValueError("Пустые данные!")

            first_item = inner_data[0] if inner_data else {}

            if "question" in first_item and "answer" in first_item:
                logger.info("Обнаружен формат: Simple Q&A")
                return DatasetFormat.SIMPLE_QA

            if "category" in first_item and "intents" in first_item:
                logger.info("Обнаружен формат: Category + Intents")
                return DatasetFormat.CATEGORY_INTENTS

            if "intentions" in first_item and "variants" in first_item and "good_answer" in first_item:
                logger.info("Обнаружен формат: Advanced Q&A")
                return DatasetFormat.ADVANCED_QA

            logger.error(f"Неизвестный формат, первый элемент: {first_item}")
            raise ValueError("Неизвестный формат данных!")

        except Exception as e:
            logger.error(f"Ошибка определения формата: {str(e)}")
            raise

    def parse_simple_qa(self, data: Dict[str, Any]) -> List[TrainingExample]:
        """Парсинг простого Q&A формата"""
        examples = []
        for item in data.get("data", [])[:self.max_examples_per_file]:
            question = self.clean_text(item.get("question", ""))
            answer = self.clean_text(item.get("answer", ""))
            if not question or not answer:
                logger.warning(f"Пропущен некорректный элемент: {item}")
                continue
            examples.append(TrainingExample(
                question=question,
                answer=answer,
                confidence=1.0,
                metadata={"source_format": "simple_qa", "source_file": str(self.data_dir)}
            ))
        logger.info(f"Обработано {len(examples)} примеров из Simple Q&A")
        return examples

    def parse_category_intents(self, data: List[Dict[str, Any]]) -> List[TrainingExample]:
        """Парсинг формата с категориями и intents"""
        examples = []
        for item in data[:self.max_examples_per_file]:
            category = self.clean_text(item.get("category", "general"))
            answer = self.clean_text(item.get("answer", ""))
            intents = item.get("intents", [])
            if not answer or not intents:
                logger.warning(f"Пропущен элемент: {item}")
                continue
            for intent in intents:
                intent = self.clean_text(intent)
                if not intent:
                    continue
                examples.append(TrainingExample(
                    question=intent,
                    answer=answer,
                    category=category,
                    confidence=0.9,
                    metadata={"source_format": "category_intents", "original_category": category}
                ))
        logger.info(f"Обработано {len(examples)} примеров из Category + Intents")
        return examples

    def parse_advanced_qa(self, data: List[Dict[str, Any]]) -> List[TrainingExample]:
        """Парсинг продвинутого формата с good/bad answers"""
        examples = []
        for item in data[:self.max_examples_per_file]:
            intentions = self.clean_text(item.get("intentions", ""))
            variants = item.get("variants", [])
            good_answer = self.clean_text(item.get("good_answer", ""))
            bad_answers = item.get("bad_answers", [])
            notes = item.get("notes", {})
            if not good_answer or not variants:
                logger.warning(f"Пропущен элемент: {item}")
                continue
            for variant in variants:
                variant = self.clean_text(variant)
                if not variant:
                    continue
                examples.append(TrainingExample(
                    question=variant,
                    answer=good_answer,
                    category=intentions,
                    confidence=1.0,
                    metadata={
                        "source_format": "advanced_qa",
                        "type": "positive",
                        "user_style": notes.get("user_style"),
                        "assistant_style": notes.get("assistant_style"),
                        "bad_patterns": notes.get("bad_pattern_types", [])
                    }
                ))
            for bad_answer in bad_answers[:1]:
                bad_answer = self.clean_text(bad_answer)
                if not bad_answer or not variants:
                    continue
                examples.append(TrainingExample(
                    question=self.clean_text(variants[0]),
                    answer=f"[NEGATIVE] {bad_answer}",
                    category=intentions,
                    confidence=0.1,
                    metadata={
                        "source_format": "advanced_qa",
                        "type": "negative",
                        "original_bad_answer": bad_answer
                    }
                ))
        logger.info(f"Обработано {len(examples)} примеров из Advanced Q&A")
        return examples

    def load_file(self, file_path: Path) -> List[TrainingExample]:
        """Загрузка и парсинг одного JSON файла"""
        logger.info(f"Загружаем файл: {file_path}")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            logger.error(f"Ошибка декодирования JSON в {file_path}: {e}")
            return []
        except Exception as e:
            logger.error(f"Ошибка загрузки {file_path}: {e}")
            return []

        try:
            format_type = self.detect_format(data)
            inner_data = data["data"] if isinstance(data, dict) and "data" in data else data
            if format_type == DatasetFormat.SIMPLE_QA:
                return self.parse_simple_qa(data)
            elif format_type == DatasetFormat.CATEGORY_INTENTS:
                return self.parse_category_intents(inner_data)
            elif format_type == DatasetFormat.ADVANCED_QA:
                return self.parse_advanced_qa(inner_data)
        except Exception as e:
            logger.error(f"Ошибка парсинга {file_path}: {e}")
            return []
        return []

    def load_all_files(self) -> 'FAQDatasetParser':
        """Загрузка всех JSON файлов с прогресс-баром"""
        json_files = list(self.data_dir.glob("*.json"))
        if not json_files:
            logger.warning(f"JSON файлы не найдены в {self.data_dir}")
            return self
        logger.info(f"Найдено {len(json_files)} JSON файлов")
        for json_file in tqdm(json_files, desc="Обработка файлов"):
            examples = self.load_file(json_file)
            self.examples.extend(examples)
        logger.info(f"Всего загружено {len(self.examples)} обучающих примеров")
        return self

    def filter_examples(self, 
                       min_question_length: int = 5,
                       min_answer_length: int = 10,
                       min_confidence: float = 0.3,
                       exclude_negative: bool = True) -> 'FAQDatasetParser':
        """Фильтрация примеров по качеству"""
        initial_count = len(self.examples)
        filtered_examples = []
        for example in self.examples:
            if len(example.question) < min_question_length:
                continue
            if len(example.answer) < min_answer_length:
                continue
            if example.confidence < min_confidence:
                continue
            if exclude_negative and "[NEGATIVE]" in example.answer:
                continue
            filtered_examples.append(example)
        self.examples = filtered_examples
        logger.info(f"Отфильтровано: {initial_count} -> {len(self.examples)} примеров")
        return self

    def get_statistics(self) -> Dict[str, Any]:
        """Расширенная статистика по датасету"""
        if not self.examples:
            return {"total": 0}
        categories = {}
        question_lengths = []
        answer_lengths = []
        confidences = []
        sources = {}
        for example in self.examples:
            cat = example.category or "uncategorized"
            categories[cat] = categories.get(cat, 0) + 1
            question_lengths.append(len(example.question))
            answer_lengths.append(len(example.answer))
            confidences.append(example.confidence)
            source = example.metadata.get("source_format", "unknown") if example.metadata else "unknown"
            sources[source] = sources.get(source, 0) + 1
        return {
            "total_examples": len(self.examples),
            "categories": categories,
            "source_formats": sources,
            "avg_question_length": sum(question_lengths) / len(question_lengths) if question_lengths else 0,
            "avg_answer_length": sum(answer_lengths) / len(answer_lengths) if answer_lengths else 0,
            "avg_confidence": sum(confidences) / len(confidences) if confidences else 0,
            "question_length_range": (min(question_lengths), max(question_lengths)) if question_lengths else (0, 0),
            "answer_length_range": (min(answer_lengths), max(answer_lengths)) if answer_lengths else (0, 0)
        }

    def export_for_training(self, output_file: Path, format_type: str = "conversational") -> bool:
        """Экспорт данных для обучения"""
        if not self.examples:
            logger.error("Нет данных для экспорта")
            return False
        training_data = []
        for example in self.examples:
            if format_type == "conversational":
                conversation = {
                    "messages": [
                        {"role": "user", "content": example.question},
                        {"role": "assistant", "content": example.answer}
                    ],
                    "metadata": {
                        "category": example.category,
                        "confidence": example.confidence,
                        "source_format": example.metadata.get("source_format") if example.metadata else None
                    }
                }
                training_data.append(conversation)
            elif format_type == "instruction":
                instruction_data = {
                    "instruction": "Ответь на вопрос студента УрФУ:",
                    "input": example.question,
                    "output": example.answer,
                    "category": example.category,
                    "confidence": example.confidence
                }
                training_data.append(instruction_data)
        try:
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(training_data, f, ensure_ascii=False, indent=2)
            logger.info(f"Данные экспортированы в {output_file}")
            return True
        except Exception as e:
            logger.error(f"Ошибка экспорта: {e}")
            return False

    def split_dataset(self, train_ratio: float = 0.8, seed: int = 42) -> Tuple[List[TrainingExample], List[TrainingExample]]:
        """Разделение на train/validation с фиксированным seed"""
        import random
        random.seed(seed)
        shuffled = self.examples.copy()
        random.shuffle(shuffled)
        split_idx = int(len(shuffled) * train_ratio)
        train_set = shuffled[:split_idx]
        val_set = shuffled[split_idx:]
        logger.info(f"Разделение: train={len(train_set)}, val={len(val_set)}")
        return train_set, val_set

def main():
    """Запуск парсера"""
    try:
        parser = FAQDatasetParser()
        parser.load_all_files()
        parser.filter_examples(min_question_length=10, min_answer_length=15, exclude_negative=True)
        stats = parser.get_statistics()
        print("\n=== СТАТИСТИКА ДАТАСЕТА ===")
        for key, value in stats.items():
            print(f"{key}: {value}")
        output_dir = Path("data/processed")
        parser.export_for_training(output_dir / "faq_conversational.json", format_type="conversational")
        parser.export_for_training(output_dir / "faq_instruction.json", format_type="instruction")
        train_examples, val_examples = parser.split_dataset(train_ratio=0.85)
        print(f"\nФинальный результат:")
        print(f"Тренировочных примеров: {len(train_examples)}")
        print(f"Валидационных примеров: {len(val_examples)}")
    except Exception as e:
        logger.error(f"Ошибка в main: {e}")
        raise

if __name__ == "__main__":
    main()