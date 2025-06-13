import json
import logging
from typing import List, Dict, Any, Union
from pathlib import Path

logger = logging.getLogger(__name__)


class FAQParser:
    """Парсер для JSON файлов FAQ двух форматов"""
    
    def __init__(self, data_dir: str = "data/faq"):
        self.data_dir = Path(data_dir)
        self.parsed_data: List[Dict[str, Any]] = []
    
    def load_json_file(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """Загружает JSON файл"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Ошибка загрузки файла {file_path}: {e}")
            return {}
    
    def parse_format_1(self, data: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        Парсит формат 1: category + intents + answer
        {
          "data": [
            {
              "category": "аттестат",
              "intents": ["вопрос1", "вопрос2"],
              "answer": "ответ"
            }
          ]
        }
        """
        parsed = []
        
        if 'data' not in data:
            logger.warning("Формат 1: Нет ключа 'data'")
            return parsed
        
        for item in data['data']:
            if not all(key in item for key in ['category', 'intents', 'answer']):
                logger.warning(f"Пропущен элемент: отсутствуют обязательные поля {item}")
                continue
            
            # Создаем отдельную запись для каждого intent
            for intent in item['intents']:
                parsed.append({
                    'question': intent.strip(),
                    'answer': item['answer'].strip(),
                    'category': item['category'].strip()
                })
        
        return parsed
    
    def parse_format_2(self, data: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        Парсит формат 2: question + answer
        {
          "data": [
            {
              "question": "вопрос",
              "answer": "ответ"
            }
          ]
        }
        """
        parsed = []
        
        if 'data' not in data:
            logger.warning("Формат 2: Нет ключа 'data'")
            return parsed
        
        for item in data['data']:
            if not all(key in item for key in ['question', 'answer']):
                logger.warning(f"Пропущен элемент: отсутствуют обязательные поля {item}")
                continue
            
            parsed.append({
                'question': item['question'].strip(),
                'answer': item['answer'].strip(),
                'category': 'general'  # Дефолтная категория
            })
        
        return parsed
    
    def detect_format(self, data: Dict[str, Any]) -> int:
        """Определяет формат JSON файла"""
        if 'data' not in data or not data['data']:
            return 0  # Неизвестный формат
        
        first_item = data['data'][0]
        
        # Формат 1: есть category, intents, answer
        if all(key in first_item for key in ['category', 'intents', 'answer']):
            return 1
        
        # Формат 2: есть question, answer
        if all(key in first_item for key in ['question', 'answer']):
            return 2
        
        return 0  # Неизвестный формат
    
    def parse_file(self, file_path: Union[str, Path]) -> List[Dict[str, str]]:
        """Парсит один JSON файл"""
        logger.info(f"Парсинг файла: {file_path}")
        
        data = self.load_json_file(file_path)
        if not data:
            return []
        
        format_type = self.detect_format(data)
        
        if format_type == 1:
            logger.info("Обнаружен формат 1 (category + intents)")
            return self.parse_format_1(data)
        elif format_type == 2:
            logger.info("Обнаружен формат 2 (question + answer)")
            return self.parse_format_2(data)
        else:
            logger.error(f"Неизвестный формат файла: {file_path}")
            return []
    
    def parse_all_files(self) -> List[Dict[str, str]]:
        """Парсит все JSON файлы в папке data/faq"""
        all_data = []
        
        if not self.data_dir.exists():
            logger.error(f"Папка {self.data_dir} не существует")
            return all_data
        
        json_files = list(self.data_dir.glob("*.json"))
        
        if not json_files:
            logger.warning(f"Не найдено JSON файлов в {self.data_dir}")
            return all_data
        
        for json_file in json_files:
            file_data = self.parse_file(json_file)
            all_data.extend(file_data)
            logger.info(f"Загружено {len(file_data)} записей из {json_file.name}")
        
        self.parsed_data = all_data
        logger.info(f"Всего загружено {len(all_data)} записей FAQ")
        
        return all_data
    
    def get_training_data(self) -> List[Dict[str, str]]:
        """Возвращает данные для обучения модели"""
        if not self.parsed_data:
            self.parse_all_files()
        
        return self.parsed_data
    
    def get_categories(self) -> List[str]:
        """Возвращает список всех категорий"""
        if not self.parsed_data:
            self.parse_all_files()
        
        categories = list(set(item['category'] for item in self.parsed_data))
        return sorted(categories)
    
    def export_to_json(self, output_path: str) -> None:
        """Экспортирует объединенные данные в JSON файл"""
        if not self.parsed_data:
            self.parse_all_files()
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'data': self.parsed_data,
                    'total_records': len(self.parsed_data),
                    'categories': self.get_categories()
                }, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Данные экспортированы в {output_path}")
        except Exception as e:
            logger.error(f"Ошибка экспорта: {e}")


if __name__ == "__main__":
    # Настройка логирования
    logging.basicConfig(level=logging.INFO)
    
    # Тест парсера
    parser = FAQParser()
    data = parser.parse_all_files()
    
    print(f"Загружено записей: {len(data)}")
    print(f"Категории: {parser.get_categories()}")
    
    if data:
        print("\nПример записи:")
        print(data[0])
    
    # Экспорт объединенных данных
    parser.export_to_json("data/faq/combined_faq.json")