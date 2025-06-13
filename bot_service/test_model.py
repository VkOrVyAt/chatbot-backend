import logging
import json
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime

from core.model_trainer import FAQModelTrainer
from core.faq_parser import FAQParser

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelTester:
    """Класс для тестирования обученной FAQ модели"""

    def __init__(self, model_path: str = None):
        if model_path is None:
        # Получаем путь к текущему файлу и строим путь к модели
            current_dir = Path(__file__).parent
            model_path = current_dir / "models" / "faq_model"
        self.model_path = Path(model_path)
        self.trainer: Optional[FAQModelTrainer] = None
        self.test_questions: List[str] = []

    def check_model_files(self) -> bool:
        """Проверяет наличие файлов модели"""
        if not self.model_path.exists():
            logger.error(f"Папка модели не существует: {self.model_path}")
            return False
        
        # Список возможных файлов модели
        model_files = [
            "model.safetensors",
            "pytorch_model.bin", 
            "pytorch_model.safetensors"
        ]
        
        # Обязательные файлы
        required_files = [
            "config.json",
            "tokenizer_config.json"
        ]
        
        found_model_file = False
        missing_files = []
        
        # Проверяем файлы модели
        for file in model_files:
            if (self.model_path / file).exists():
                found_model_file = True
                logger.info(f"Найден файл модели: {file}")
                break
        
        # Проверяем обязательные файлы
        for file in required_files:
            if not (self.model_path / file).exists():
                missing_files.append(file)
        
        if not found_model_file:
            logger.error("Не найден ни один файл модели!")
            self.show_directory_contents()
            return False
        
        if missing_files:
            logger.error(f"Отсутствуют обязательные файлы: {missing_files}")
            self.show_directory_contents()
            return False
        
        return True

    def show_directory_contents(self) -> None:
        """Показывает содержимое папки модели для диагностики"""
        print(f"\n🔍 Диагностика папки модели: {self.model_path}")
        print("=" * 80)
        
        if not self.model_path.exists():
            print("❌ Папка модели не существует!")
            print(f"Ожидаемый путь: {self.model_path}")
            print("\n💡 Решение:")
            print("   Сначала обучите модель: python -m core.model_trainer")
            return
        
        files = list(self.model_path.glob("*"))
        if not files:
            print("❌ Папка модели пуста!")
            print("\n💡 Решение:")
            print("   Запустите обучение модели: python -m core.model_trainer")
            return
        
        print("📁 Содержимое папки:")
        for file in sorted(files):
            if file.is_file():
                size = f"{file.stat().st_size:,} bytes"
            else:
                size = "DIR"
            print(f"  - {file.name} ({size})")
        
        # Проверяем ключевые файлы
        print("\n🔧 Проверка файлов:")
        
        config_files = ["config.json", "tokenizer_config.json", "tokenizer.json"]
        model_files = ["model.safetensors", "pytorch_model.bin", "pytorch_model.safetensors"]
        
        print("  Конфигурационные файлы:")
        for file in config_files:
            exists = (self.model_path / file).exists()
            status = "✅" if exists else "❌"
            print(f"    {status} {file}")
        
        print("  Файлы модели:")
        model_exists = False
        for file in model_files:
            exists = (self.model_path / file).exists()
            status = "✅" if exists else "❌"
            print(f"    {status} {file}")
            if exists:
                model_exists = True
        
        if not model_exists:
            print("\n❌ ПРОБЛЕМА: Не найден ни один файл модели!")
            print("💡 Рекомендации:")
            print("   1. Переобучите модель: python -m core.model_trainer")
            print("   2. Проверьте логи обучения на наличие ошибок")
            print("   3. Убедитесь, что обучение завершилось успешно")

    def load_model(self) -> bool:
        """Загружает обученную модель"""
        try:
            # Сначала проверяем наличие всех необходимых файлов
            if not self.check_model_files():
                return False

            logger.info(f"Загрузка модели из {self.model_path}")
            self.trainer = FAQModelTrainer(output_dir=str(self.model_path))
            self.trainer.load_trained_model()
            logger.info("Модель успешно загружена")
            return True

        except FileNotFoundError as e:
            logger.error(f"Файл модели не найден: {e}")
            self.show_directory_contents()
            return False
        except Exception as e:
            logger.exception("Ошибка загрузки модели")
            self.show_directory_contents()
            return False

    def prepare_test_questions(self) -> None:
        """Подготавливает тестовые вопросы"""
        self.test_questions = [
            "Как поступить в UrFU?",
            "Какие факультеты есть в университете?",
            "Где находится университет?",
            "Сколько стоит обучение?",
            "Какие документы нужны для поступления?",
            "Есть ли общежитие?",
            "Какие есть кружки и секции?",
            "Как получить стипендию?",
            "Где можно поесть в университете?",
            "Есть ли библиотека?",
            "Как восстановить пароль от личного кабинета?",
            "Где скачать справку об обучении?",
            "Как записаться на пересдачу?",
            "Когда начинается учебный год?",
            "Как связаться с деканатом?",
            "Какие направления подготовки есть на ИнФО?",
            "Что такое модульно-рейтинговая система?",
            "Как получить академический отпуск?",
            "Есть ли возможность обучения по обмену?",
            "Какие языки можно изучать?",
        ]

    def test_single_question(self, question: str, max_tokens: int = 150) -> Dict[str, str]:
        """Тестирует одиночный вопрос"""
        if not self.trainer:
            return {"question": question, "answer": "Модель не загружена", "status": "error"}

        try:
            logger.info(f"Тестирование вопроса: {question}")
            answer = self.trainer.generate_answer(question, max_new_tokens=max_tokens)
            return {"question": question, "answer": answer, "status": "success"}
        except Exception as e:
            logger.exception(f"Ошибка генерации ответа для вопроса: {question}")
            return {"question": question, "answer": f"Ошибка: {str(e)}", "status": "error"}

    def test_all_questions(self, save_results: bool = True) -> List[Dict[str, str]]:
        """Тестирует все подготовленные вопросы"""
        if not self.test_questions:
            self.prepare_test_questions()

        results = []
        logger.info(f"Начинаем тестирование {len(self.test_questions)} вопросов")

        for i, question in enumerate(self.test_questions, 1):
            print(f"\n{'=' * 80}")
            print(f"ВОПРОС {i}: {question}")
            result = self.test_single_question(question)
            print(f"ОТВЕТ: {result['answer']}")
            print(f"СТАТУС: {result['status']}")
            print(f"{'=' * 80}")
            results.append(result)

        if save_results:
            self.save_test_results(results)

        return results

    def save_test_results(self, results: List[Dict[str, str]]) -> None:
        """Сохраняет результаты тестирования"""
        try:
            output_file = self.model_path / "test_results.json"
            test_data = {
                "model_path": str(self.model_path),
                "total_questions": len(results),
                "successful_answers": sum(r["status"] == "success" for r in results),
                "failed_answers": sum(r["status"] == "error" for r in results),
                "test_timestamp": datetime.now().isoformat(),
                "results": results
            }

            with output_file.open("w", encoding="utf-8") as f:
                json.dump(test_data, f, ensure_ascii=False, indent=2)

            logger.info(f"Результаты тестирования сохранены в {output_file}")
        except Exception as e:
            logger.exception("Ошибка при сохранении результатов тестирования")

    def interactive_test(self) -> None:
        """Интерактивное тестирование модели"""
        if not self.trainer:
            logger.error("Модель не загружена")
            return

        print("\n" + "=" * 80)
        print("ИНТЕРАКТИВНОЕ ТЕСТИРОВАНИЕ МОДЕЛИ FAQ")
        print("=" * 80)

        while True:
            try:
                question = input("\nВаш вопрос (или 'quit' для выхода): ").strip()
                if question.lower() in ['quit', 'exit', 'выход']:
                    print("Выход из режима тестирования...")
                    break
                if not question:
                    print("Пожалуйста, введите вопрос.")
                    continue
                result = self.test_single_question(question)
                print(f"\n{'-' * 60}")
                print(f"Ответ модели: {result['answer']}")
                print(f"{'-' * 60}")
            except KeyboardInterrupt:
                print("\n\nТестирование прервано пользователем.")
                break

    def compare_with_original_faq(self) -> None:
        """Сравнивает ответы модели с оригинальными FAQ"""
        if not self.trainer:
            logger.error("Модель не загружена")
            return

        try:
            parser = FAQParser()
            faq_data = parser.parse_all_files()
            if not faq_data:
                logger.error("Не удалось загрузить оригинальные FAQ")
                return

            print("\n" + "=" * 100)
            print("СРАВНЕНИЕ С ОРИГИНАЛЬНЫМИ FAQ")
            print("=" * 100)

            for i, faq_item in enumerate(faq_data[:10], 1):
                question = faq_item['question']
                original_answer = faq_item['answer']
                result = self.test_single_question(question)
                model_answer = result["answer"]

                print(f"\n{'-' * 100}")
                print(f"Сравнение {i}")
                print(f"Вопрос: {question}")
                print(f"\nОригинальный ответ:\n{original_answer}")
                print(f"\nОтвет модели:\n{model_answer}")
                print(f"{'-' * 100}")

        except Exception as e:
            logger.exception("Ошибка при сравнении с оригинальными FAQ")

    def get_model_info(self) -> Dict:
        """Получает информацию о загруженной модели"""
        try:
            config_path = self.model_path / "training_config.json"
            if config_path.exists():
                with config_path.open("r", encoding="utf-8") as f:
                    return json.load(f)
            return {"error": "Файл конфигурации не найден"}
        except Exception as e:
            logger.exception("Ошибка чтения конфигурации модели")
            return {"error": str(e)}

    def diagnose_model(self) -> None:
        """Диагностика состояния модели"""
        print("\n" + "=" * 80)
        print("ДИАГНОСТИКА МОДЕЛИ")
        print("=" * 80)
        
        self.show_directory_contents()
        
        # Проверяем размеры файлов
        if self.model_path.exists():
            print("\n📊 Размеры ключевых файлов:")
            key_files = ["model.safetensors", "pytorch_model.bin", "config.json", "tokenizer_config.json"]
            for file_name in key_files:
                file_path = self.model_path / file_name
                if file_path.exists():
                    size_mb = file_path.stat().st_size / (1024 * 1024)
                    print(f"  {file_name}: {size_mb:.2f} MB")


def main():
    tester = ModelTester()
    
    print("🤖 FAQ Model Tester")
    print("=" * 50)
    
    # Сначала показываем диагностику
    tester.diagnose_model()
    
    if not tester.load_model():
        print("\n❌ Не удалось загрузить модель.")
        print("💡 Убедитесь, что модель обучена: python -m core.model_trainer")
        return

    model_info = tester.get_model_info()
    print("\n" + "=" * 80)
    print("ИНФОРМАЦИЯ О МОДЕЛИ")
    print("=" * 80)
    for key, value in model_info.items():
        print(f"{key}: {value}")
    print("=" * 80)

    while True:
        print("\nВыберите режим:")
        print("1. Автоматическое тестирование")
        print("2. Интерактивное тестирование")
        print("3. Сравнение с оригинальными FAQ")
        print("4. Тест одного вопроса")
        print("5. Диагностика модели")
        print("6. Выход")

        choice = input("Ваш выбор (1-6): ").strip()

        if choice == "1":
            tester.test_all_questions()
        elif choice == "2":
            tester.interactive_test()
        elif choice == "3":
            tester.compare_with_original_faq()
        elif choice == "4":
            q = input("Введите вопрос: ").strip()
            if q:
                result = tester.test_single_question(q)
                print(f"\nОтвет: {result['answer']}")
        elif choice == "5":
            tester.diagnose_model()
        elif choice == "6":
            print("Выход.")
            break
        else:
            print("Неверный выбор. Попробуйте снова.")


if __name__ == "__main__":
    main()