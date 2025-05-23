import json
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, AutoTokenizer
from transformers import DataCollatorForLanguageModeling

# Загрузите токенизатор
tokenizer = AutoTokenizer.from_pretrained('distilgpt2')  # Используйте вашу модель

class MultiFAQDataset(Dataset):
    def __init__(self,
                 json_paths: list[str],
                 tokenizer: PreTrainedTokenizer,
                 max_length: int = 256):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.items: list[tuple[str, str]] = []

        for path in json_paths:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f).get("data", [])
            for entry in data:
                answer = entry.get("answer", "").strip()
                if not answer:
                    continue
                if isinstance(entry.get("intents"), list):
                    for intent in entry["intents"]:
                        self.items.append((intent.strip(), answer))
                elif "question" in entry:
                    self.items.append((entry["question"].strip(), answer))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        intent, answer = self.items[idx]
        prompt = f"Вопрос: {intent}\nОтвет:"

        # Получаем списки токенов через encode с явным паддингом
        prompt_ids: list[int] = self.tokenizer.encode(
            prompt,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_length,
            padding='max_length'  # Добавляем паддинг до максимальной длины
        )

        # Оставшийся лимит для ответа
        remain = self.max_length - len(prompt_ids) - 1
        answer_ids: list[int] = self.tokenizer.encode(
            answer,
            add_special_tokens=False,
            truncation=True,
            max_length=max(0, remain),
            padding='max_length'  # Добавляем паддинг до максимальной длины
        )

        # Собираем input_ids: prompt + answer + eos
        eos_id = self.tokenizer.eos_token_id
        input_ids = prompt_ids + answer_ids + [eos_id]
        if len(input_ids) > self.max_length:
            input_ids = input_ids[: self.max_length]

        # labels: -100 для prompt, реальные токены для ответа и eos
        labels = [-100] * len(prompt_ids) + input_ids[len(prompt_ids):]
        labels = labels[: self.max_length]

        # attention_mask: единицы для всех токенов
        attention_mask = [1] * len(input_ids)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

# Используйте padding=True в DataCollatorForLanguageModeling
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
    pad_to_multiple_of=None  # Если нужно, добавьте это
)
