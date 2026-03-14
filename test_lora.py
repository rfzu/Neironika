#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

BASE_MODEL = "huihui-ai/Llama-3.2-3B-Instruct-abliterated"
ADAPTER_DIR = "./svp_lora_verbose"

TEST_PROMPTS = [
    "Расскажи кратко, что такое системно-векторная психология.",
    "Какие свойства обычно приписывают кожному вектору?",
    "Чем зрительный вектор отличается от звукового?",
    "Как в СВП объясняют детские страхи?",
    "Дай структурированный ответ о связке анального и кожного векторов.",
]


def build_prompt(user_text: str) -> str:
    return (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
        "Ты полезный ассистент и эксперт по системно-векторной психологии.<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n"
        f"{user_text}<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n"
    )


def main():
    if not os.path.exists(ADAPTER_DIR):
        raise FileNotFoundError(f"Не найдена папка адаптера: {ADAPTER_DIR}")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA недоступна")

    print("=" * 80)
    print("🧪 TEST LORA")
    print("=" * 80)

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
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
    )

    print("Подключение LoRA адаптера...")
    model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
    model.eval()

    for i, prompt in enumerate(TEST_PROMPTS, 1):
        full_prompt = build_prompt(prompt)
        inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=220,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        decoded = tokenizer.decode(outputs[0], skip_special_tokens=False)
        answer = decoded[len(full_prompt):]
        answer = answer.replace("<|eot_id|>", "").strip()

        print("\n" + "=" * 80)
        print(f"ТЕСТ #{i}")
        print(f"ВОПРОС: {prompt}")
        print("-" * 80)
        print(answer)
        print("=" * 80)

if __name__ == "__main__":
    main()
