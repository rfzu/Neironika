#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import gc
import logging
import os
import sys
import time
from datetime import datetime

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainerCallback,
    TrainingArguments,
)
from trl import SFTTrainer

MODEL_NAME = "huihui-ai/Llama-3.2-3B-Instruct-abliterated"
DATA_FILE = "merged_dataset_formatted.jsonl"
OUTPUT_DIR = "./svp_lora_verbose"
LOG_FILE = "train.log"


def setup_logging():
    logger = logging.getLogger("svp_lora")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


logger = setup_logging()


def log(msg):
    logger.info(msg)


class VerboseTrainerCallback(TrainerCallback):
    def __init__(self):
        self.start_time = time.time()

    def on_train_begin(self, args, state, control, **kwargs):
        log("🚀 Обучение началось")
        log(f"Всего шагов: {state.max_steps}")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return

        elapsed = time.time() - self.start_time
        steps_per_min = state.global_step / elapsed * 60 if elapsed > 0 and state.global_step > 0 else 0
        vram_alloc = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
        vram_reserved = torch.cuda.memory_reserved() / 1024**3 if torch.cuda.is_available() else 0

        parts = [
            f"step={state.global_step}",
            f"epoch={state.epoch:.2f}" if state.epoch is not None else "epoch=?",
        ]

        if "loss" in logs:
            parts.append(f"loss={logs['loss']:.4f}")
        if "learning_rate" in logs:
            parts.append(f"lr={logs['learning_rate']:.2e}")
        if "grad_norm" in logs:
            parts.append(f"grad_norm={logs['grad_norm']:.4f}")

        parts.append(f"steps/min={steps_per_min:.2f}")
        parts.append(f"vram_alloc={vram_alloc:.2f}GB")
        parts.append(f"vram_reserved={vram_reserved:.2f}GB")

        log(" | ".join(parts))

    def on_save(self, args, state, control, **kwargs):
        log(f"💾 Чекпоинт сохранён на шаге {state.global_step}")

    def on_train_end(self, args, state, control, **kwargs):
        elapsed = time.time() - self.start_time
        log(f"✅ Обучение завершено за {elapsed / 60:.1f} мин")


def main():
    start_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log("=" * 80)
    log("🧠 SVP LoRA TRAINER")
    log(f"Старт: {start_ts}")
    log("=" * 80)

    if not os.path.exists(DATA_FILE):
        log(f"❌ Не найден файл датасета: {DATA_FILE}")
        raise FileNotFoundError(DATA_FILE)

    log("1/7 Загрузка токенизатора...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    log("✅ Токенизатор загружен")

    log("2/7 Настройка 4-bit квантизации...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    log("✅ Конфиг bitsandbytes готов")

    log("3/7 Загрузка базовой модели...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    model = prepare_model_for_kbit_training(model)
    log("✅ Базовая модель загружена и подготовлена")

    log("4/7 Применение LoRA...")
    lora_config = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    log("✅ LoRA адаптеры подключены")

    log("5/7 Загрузка датасета...")
    dataset = load_dataset("json", data_files=DATA_FILE, split="train")
    log(f"✅ Датасет загружен, записей: {len(dataset)}")
    log(f"Колонки датасета: {dataset.column_names}")

    if "text" not in dataset.column_names:
        raise ValueError("В датасете нет колонки 'text'")

    sample = dataset[0]["text"][:200].replace("\n", "\\n")
    log(f"Пример текста: {sample}...")

    log("6/7 Настройка TrainingArguments...")
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        num_train_epochs=3,
        learning_rate=1e-4,
        warmup_steps=10,
        logging_steps=5,
        save_steps=50,
        save_total_limit=2,
        fp16=True,
        dataloader_num_workers=0,
        report_to=[],
        optim="paged_adamw_8bit",
        remove_unused_columns=False,
        logging_dir="./logs",
    )
    log("✅ TrainingArguments готовы")

    log("7/7 Создание SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    trainer.add_callback(VerboseTrainerCallback())
    log("✅ SFTTrainer создан")

    log("🎓 Запуск trainer.train() ...")
    trainer.train()

    log("💾 Сохранение финального адаптера...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    log(f"✅ Модель и токенизатор сохранены в {OUTPUT_DIR}")

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    log("🧹 Память очищена")
    log("Готово")


if __name__ == "__main__":
    main()
