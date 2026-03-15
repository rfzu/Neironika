#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import torch
import traceback
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

BASE_MODEL = "huihui-ai/Llama-3.2-3B-Instruct-abliterated"
ADAPTER_DIR = "./svp_lora_verbose"

SYSTEM_PROMPT = (
    "Ты полезный психолог Нейроника Степанова и эксперт по системно-векторной психологии. "
    "Отвечай в яркой, уверенной, провокационной и разговорной манере, "
    "Давай психологические разборы смело, прямо и образно, но по делу. "
    "Фокусируйся на мотивах, страхах, желаниях, самооценке, отношениях, "
    "внутренних конфликтах и поведенческих сценариях. "

    "Структура ответа: "
    "1️⃣ Определение вектора и его роль в стае "
    "2️⃣ Основные качества и сценарии поведения "
    "3️⃣ Как проявляется в отношениях/работе/жизни "
    "4️⃣ Практический совет или вывод "
    
    "Используй образный язык: 'кожник как снайпер', 'анальник как музейный хранитель'. "
    "Будь конкретен: свойства, сценарии, типичные проблемы."
)

# Настройки генерации
MAX_NEW_TOKENS = 300
USE_SAMPLING = True
TEMPERATURE = 0.7
TOP_P = 0.9
REPETITION_PENALTY = 1.1


def load_model():
    if not os.path.exists(ADAPTER_DIR):
        raise FileNotFoundError(f"Не найдена папка адаптера: {ADAPTER_DIR}")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA недоступна")

    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    print("Загрузка базовой модели...")
    print(f"Using model: {BASE_MODEL}")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
    )

    print("Подключение LoRA адаптера...")
    model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
    model.eval()

    return tokenizer, model


def generate_answer(tokenizer, model, messages):
    enc = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )

    input_ids = enc["input_ids"].to(model.device)
    attention_mask = enc.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(model.device)

    gen_kwargs = dict(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=MAX_NEW_TOKENS,
        repetition_penalty=REPETITION_PENALTY,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    if USE_SAMPLING:
        gen_kwargs.update(
            dict(
                do_sample=True,
                temperature=TEMPERATURE,
                top_p=TOP_P,
            )
        )
    else:
        gen_kwargs.update(
            dict(
                do_sample=False,
                temperature=1.0,
                top_p=1.0,
            )
        )

    t0 = time.time()
    with torch.no_grad():
        outputs = model.generate(**gen_kwargs)
    dt = time.time() - t0
    print(f"(генерация заняла {dt:.1f} с)")

    new_tokens = outputs[0][input_ids.shape[-1]:]
    answer = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return answer


def main():
    tokenizer, model = load_model()

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]

    print("=" * 80)
    print("🧠 LoRA чат запущен")
    print("Напиши 'exit' для выхода, 'clear' чтобы очистить историю")
    print("=" * 80)

    while True:
        user_text = input("\nТы: ").strip()

        if not user_text:
            continue

        if user_text.lower() in {"exit", "quit", "q"}:
            print("Выход.")
            break

        if user_text.lower() == "clear":
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            print("История очищена.")
            continue

        messages.append({"role": "user", "content": user_text})

        try:
            answer = generate_answer(tokenizer, model, messages)
            print(f"\nМодель: {answer}")
            messages.append({"role": "assistant", "content": answer})
        except Exception as e:
            print("\n=== ОШИБКА ГЕНЕРАЦИИ ===")
            print(repr(e))
            traceback.print_exc()


if __name__ == "__main__":
    main()
