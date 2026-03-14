#!/usr/bin/env python3
"""
🚀 SVP LoRA Trainer — подробные логи обучения
1110 примеров → эксперт по системно-векторной психологии
"""

import torch
import time
from datetime import datetime
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    TrainingArguments, 
    BitsAndBytesConfig  # ← вот этого не хватало
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from datasets import load_dataset
import gc


print("=" * 70)
print("🧠 SVP LoRA TRAINER — 1110 примеров")
print(f"⏰ Started: {datetime.now().strftime('%H:%M:%S')}")
print("=" * 70)

# Шаг 1
print("\n📥 1/7 Загрузка модели и токенизатора...")
model_name = "huihui-ai/Llama-3.2-3B-Instruct-abliterated"
print(f"   Модель: {model_name}")
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
print("   ✅ Токенизатор готов")

# Шаг 2
print("\n⚙️  2/7 4-bit квантизация для RTX 1080...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
)
model = prepare_model_for_kbit_training(model)
print("   ✅ Модель загружена (4-bit)")

# Шаг 3  
print("\n🔧 3/7 LoRA адаптеры...")
lora_config = LoraConfig(
    r=32, lora_alpha=64,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none", task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
print("   ✅ LoRA готов (32 rank)")

# Шаг 4
print("\n📚 4/7 Загрузка датасета...")
dataset = load_dataset("json", data_files="merged_dataset_formatted.jsonl", split="train")
print(f"   📊 Примеров: {len(dataset)}")
print(f"   📝 Средняя длина: {dataset[0]['text'][:100]}...")

def formatting_prompts_func(example):
    return {"text": example["text"]}
dataset = dataset.map(formatting_prompts_func)
print("   ✅ Датасет готов")

# Шаг 5
print("\n🎯 5/7 Конфигурация тренировки...")
training_args = TrainingArguments(
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    warmup_steps=10,
    num_train_epochs=3,
    learning_rate=1e-4,
    fp16=True,
    logging_steps=5,                # каждые 5 шагов
    save_steps=50,
    output_dir="./svp_lora_verbose",
    optim="paged_adamw_8bit",
    report_to=None,
    dataloader_num_workers=0,
)

trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,  # ← вот это исправление
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=2048,
    packing=False,
    args=training_args,
)


print("   ✅ Тренер настроен")

# Шаг 6 — ТРЕНИРОВКА с подробными логами
print("\n🚀 6/7 НАЧИНАЕМ ОБУЧЕНИЕ...")
print("Шаг  | Эпоха | Loss | Примеры/сек | VRAM")
print("-" * 45)

class VerboseCallback:
    def __init__(self):
        self.start_time = time.time()
    
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % 5 == 0:
            elapsed = time.time() - self.start_time
            speed = state.global_step / elapsed if elapsed > 0 else 0
            vram = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
            
            print(f"{state.global_step:4d} | {state.epoch:4.1f} | "
                  f"{state.log_history[-1]['train_loss']:5.2f} | "
                  f"{speed:9.1f} | {vram:5.1f}GB")

trainer.add_callback(VerboseCallback())
print("   Callback для логов добавлен")

# Шаг 7
print("\n🎓 7/7 ТРЕНИРУЕМ (Ctrl+C для остановки)...")
trainer.train()

print("\n" + "=" * 70)
print("✅ ОБУЧЕНИЕ ЗАВЕРШЕНО!")
print(f"⏰ Закончено: {datetime.now().strftime('%H:%M:%S')}")
print("📁 Модель: ./svp_lora_verbose")
print("\n🧪 Тест:")
print("python test_svp_expert.py")
print("=" * 70)

# Очистка памяти
gc.collect()
torch.cuda.empty_cache()
