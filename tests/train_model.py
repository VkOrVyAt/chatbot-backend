import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from dataset import MultiFAQDataset

def main():
    # Выбираем русскоязычную предобученную модель
    model_name = "ai-forever/rugpt3small_based_on_gpt2"
    print(f"Загружаем модель {model_name}...")

    # Загружаем токенизатор и модель
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Обеспечиваем совместимость токенов
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.resize_token_embeddings(len(tokenizer))

    # Загружаем датасет (все 3 JSON-файла)
    json_paths = ["tests/FAQ.json", "tests/FAQ_2.json", "tests/FAQ_3.json"]
    dataset = MultiFAQDataset(json_paths=json_paths, tokenizer=tokenizer)

    # Создаём data collator с авто-паддингом
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    # Параметры обучения
    training_args = TrainingArguments(
        output_dir="tests/model",
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        save_steps=30,
        save_total_limit=1,
        logging_steps=10,
        prediction_loss_only=True,
        report_to="none",
    )

    # Создаём Trainer (без передачи tokenizer параметром)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )

    print(f"Начинаем дообучение модели {model_name}...")
    trainer.train()

    # Сохраняем модель
    print("Сохраняем дообученную модель в tests/model")
    model.save_pretrained("tests/model")
    tokenizer.save_pretrained("tests/model")

if __name__ == "__main__":
    main()
