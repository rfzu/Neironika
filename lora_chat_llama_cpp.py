#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LoRA чат через llama.cpp (GGUF) - БЫСТРАЯ ВЕРСИЯ
Использует системный промпт для имитации стиля LoRA
"""

import os
import sys
import ctypes
import json

# === НАСТРОЙКИ ===
# Обновлённая модель с LoRA (Q8_0)
MODEL_PATH = r"D:\code\LLM_LoRA\models\svp-lora-merged-Q8_0.gguf"
LORA_DIR = None  # LoRA теперь встроена в модель

# Параметры
N_CTX = 4096
N_THREADS = 8
N_GPU_LAYERS = 35
N_BATCH = 512

# Настройки генерации (как в оригинальном lora_chat.py)
MAX_TOKENS = 300
TEMPERATURE = 0.7
TOP_P = 0.9
TOP_K = 40
REPETITION_PENALTY = 1.1

# LoRA встроена в модель - системный промпт не требуется
DEFAULT_SYSTEM_PROMPT = None


def load_system_prompt_from_lora():
    """LoRA встроена в модель - промпт не требуется"""
    return None


def setup_dll_paths():
    """Настройка путей к DLL для CUDA"""
    venv_path = os.path.dirname(sys.executable)
    dll_dirs = [
        os.path.join(venv_path, "Lib", "site-packages", "nvidia", "cublas", "bin"),
        os.path.join(venv_path, "Lib", "site-packages", "nvidia", "cuda_runtime", "bin"),
        os.path.join(venv_path, "Lib", "site-packages", "nvidia", "cudnn", "bin"),
        os.path.join(venv_path, "Lib", "site-packages", "llama_cpp", "lib"),
    ]
    os.environ["PATH"] = ";".join(dll_dirs) + ";" + os.environ.get("PATH", "")
    for dll_dir in dll_dirs:
        if os.path.exists(dll_dir):
            os.add_dll_directory(dll_dir)


def load_model():
    """Загрузка модели"""
    from llama_cpp import Llama
    
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Модель не найдена: {MODEL_PATH}")
        return None
    
    print(f"📦 Загрузка модели: {MODEL_PATH}")
    print(f"🎮 GPU слои: {N_GPU_LAYERS}")
    print(f"⚙️  Потоки CPU: {N_THREADS}")
    
    llm = Llama(
        model_path=MODEL_PATH,
        n_ctx=N_CTX,
        n_threads=N_THREADS,
        n_gpu_layers=N_GPU_LAYERS,
        n_batch=N_BATCH,
        n_ubatch=N_BATCH,
        flash_attn=True,
        verbose=False,
    )
    
    print("✅ Модель загружена\n")
    return llm


def chat_once(llm, history, user_msg, system_prompt=None, stream=True):
    """Генерация ответа с streaming"""

    messages = []
    # Добавляем системный промпт только если он задан
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    
    messages.extend(history)
    messages.append({"role": "user", "content": user_msg})
    
    print("\n🤖 Модель:\n", end="", flush=True)
    
    if stream:
        answer_parts = []
        output = llm.create_chat_completion(
            messages=messages,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            top_k=TOP_K,
            repeat_penalty=REPETITION_PENALTY,
            stream=True,
        )
        for chunk in output:
            try:
                delta_content = chunk["choices"][0]["delta"].get("content", "")
            except (KeyError, IndexError):
                delta_content = ""
            if delta_content:
                print(delta_content, end="", flush=True)
                answer_parts.append(delta_content)
        answer = "".join(answer_parts).strip()
        print()
    else:
        output = llm.create_chat_completion(
            messages=messages,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            top_k=TOP_K,
            repeat_penalty=REPETITION_PENALTY,
            stream=False,
        )
        answer = output["choices"][0]["message"]["content"].strip()
        print(answer)
    
    history.append({"role": "assistant", "content": answer})
    print("\n" + "="*80 + "\n")
    return answer


def main():
    setup_dll_paths()
    
    # Загрузка системного промпта
    system_prompt = load_system_prompt_from_lora()
    
    llm = load_model()
    if llm is None:
        return
    
    history = []
    
    print("=" * 80)
    print("🧠 LoRA чат (llama.cpp + GGUF) - Слитая модель")
    print("📚 Модель: svp-lora-merged-Q8_0.gguf (LoRA встроена)")
    print("Напиши 'exit' для выхода, 'clear' чтобы очистить историю")
    print("=" * 80)
    
    while True:
        try:
            user_text = input("\nТы: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        
        if not user_text:
            continue
        
        if user_text.lower() in {"exit", "quit", "q"}:
            print("Выход.")
            break
        
        if user_text.lower() == "clear":
            history = []
            print("История очищена.")
            continue
        
        chat_once(llm, history, user_text, system_prompt, stream=True)
    
    print("👋 Пока!")


if __name__ == "__main__":
    main()
