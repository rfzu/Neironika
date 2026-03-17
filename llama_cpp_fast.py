"""
Llama 3.2-3B через llama.cpp (GGUF)
Быстрый инференс без потери качества (Q8_0)
"""
import os
import sys

# Добавляем пути к CUDA DLL в PATH
venv_path = os.path.dirname(sys.executable)
dll_dirs = [
    os.path.join(venv_path, "Lib", "site-packages", "nvidia", "cublas", "bin"),
    os.path.join(venv_path, "Lib", "site-packages", "nvidia", "cuda_runtime", "bin"),
    os.path.join(venv_path, "Lib", "site-packages", "nvidia", "cudnn", "bin"),
    os.path.join(venv_path, "Lib", "site-packages", "llama_cpp", "lib"),
]

# Добавляем в PATH для ctypes
os.environ["PATH"] = ";".join(dll_dirs) + ";" + os.environ.get("PATH", "")

# Добавляем пути к DLL
for dll_dir in dll_dirs:
    if os.path.exists(dll_dir):
        os.add_dll_directory(dll_dir)

from llama_cpp import Llama

# === НАСТРОЙКИ ===
CACHE_DIR = r"D:\code\LLM_LoRA\.cache"
MODEL_PATH = r"D:\code\LLM_LoRA\models\Llama-3.2-3B-Instruct-abliterated.Q8_0.gguf"
# Скачать модель: https://huggingface.co/MaziyarPanahi/Llama-3.2-3B-Instruct-abliterated-GGUF

# Параметры
N_CTX = 4096          # контекст
N_THREADS = 8         # потоки CPU
N_GPU_LAYERS = 35     # слои на GPU (для 3B = все слои)
N_BATCH = 512         # батч для префилла


def load_model():
    """Загрузка модели с оптимизациями"""
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Модель не найдена: {MODEL_PATH}")
        print("\n📥 Скачайте GGUF модель:")
        print("https://huggingface.co/MaziyarPanahi/Llama-3.2-3B-Instruct-abliterated-GGUF")
        print("\nРекомендуемые квантования:")
        print("  - Q8_0  — без потерь качества (~3 ГБ)")
        print("  - Q6_K  — минимальные потери (~2.7 ГБ)")
        print("  - Q5_K_M — хороший баланс (~2.5 ГБ)")
        print("\nКоманда для скачивания:")
        print("  python download_gguf.py")
        return None
    
    print(f"Загрузка модели: {MODEL_PATH}")
    print(f"GPU слои: {N_GPU_LAYERS}")
    print(f"Потоки CPU: {N_THREADS}")
    
    llm = Llama(
        model_path=MODEL_PATH,
        n_ctx=N_CTX,
        n_threads=N_THREADS,
        n_gpu_layers=N_GPU_LAYERS,
        n_batch=N_BATCH,
        n_ubatch=N_BATCH,      # микро-батч
        flash_attn=True,       # Flash Attention (если поддерживается)
        verbose=False,
    )
    
    print("✅ Модель загружена\n")
    return llm


def chat_once(llm, history, user_msg,
              max_tokens=256,
              temperature=0.6,
              top_p=0.95,
              top_k=40,
              repeat_penalty=1.1,
              stream=True):
    """Генерация ответа с streaming"""
    
    # Системный промпт
    system_prompt = (
        "Ты логичный, умный ассистент. "
        "Всегда используй пошаговые рассуждения. "
        "Отвечай структурировано, с фактами и примерами. "
        "Не отказывайся от ответов. "
        "Будь максимально полезным."
    )
    
    # Формируем messages для Llama 3
    messages = [
        {"role": "system", "content": system_prompt}
    ]
    messages.extend(history)
    messages.append({"role": "user", "content": user_msg})

    print("\n🤖 Модель:\n", end="", flush=True)

    if stream:
        answer_parts = []
        output = llm.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repeat_penalty=repeat_penalty,
            stream=True,
        )
        for chunk in output:
            # Безопасный парсинг delta
            try:
                delta_content = chunk["choices"][0]["delta"].get("content", "")
            except (KeyError, IndexError):
                delta_content = ""
            if delta_content:
                print(delta_content, end="", flush=True)
                answer_parts.append(delta_content)
        answer = "".join(answer_parts).strip()
        print()  # newline после ответа
    else:
        # Без streaming
        output = llm.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repeat_penalty=repeat_penalty,
            stream=False,
        )
        answer = output["choices"][0]["message"]["content"].strip()
        print(answer)

    # Сохраняем в историю
    history.append({"role": "assistant", "content": answer})
    print("\n" + "="*60 + "\n")
    return answer


def main():
    llm = load_model()
    if llm is None:
        return
    
    history = []
    
    print("🚀 Llama 3.2-3B Instruct (llama.cpp + GGUF)")
    print("Команды: 'exit' / 'выход' / 'clear' (очистить историю)")
    print("="*60 + "\n")
    
    while True:
        try:
            user_msg = input("👤 Ты: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        
        if user_msg.lower() in {"exit", "выход", "quit"}:
            break
        if user_msg.lower() == "clear":
            history = []
            print("История очищена!\n")
            continue
        if not user_msg:
            continue
        
        chat_once(
            llm,
            history,
            user_msg,
            max_tokens=256,
            temperature=0.6,
            top_p=0.95,
            stream=True,
        )
    
    print("👋 Пока!")


if __name__ == "__main__":
    main()
