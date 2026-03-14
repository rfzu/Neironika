import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextStreamer,
)

# --- настройки модели ---
MODEL_NAME = "huihui-ai/Qwen2.5-Coder-3B-Instruct-abliterated"  # при желании замени на свою
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_model():
    print(f"Using device: {DEVICE}")
    print(f"Using model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype="auto",
        device_map="auto",      # сам раскинет на GPU/CPU
    ).eval()

    return tokenizer, model


def chat_once_stream(
    tokenizer,
    model,
    history,
    user_msg,
    max_new_tokens=192,    # 192 токена: достаточно информативно и ещё довольно быстро
    temperature=0.7,
    top_p=0.9,
):
    # добавляем пользовательское сообщение в историю
    history.append({"role": "user", "content": user_msg})

    # применяем chat template Qwen
    text = tokenizer.apply_chat_template(
        history,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # стриминг в консоль
    streamer = TextStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
    )

    print("\nМодель:\n", end="", flush=True)

    # генерация с рандомом (temperature/top_p) и без лишних флагов
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,                 # нужен для использования temperature/top_p
            temperature=temperature,
            top_p=top_p,
            top_k=50,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.05,
            streamer=streamer,              # стримим токены
        )

    # отдельный разбор новых токенов для history
    new_ids = generated_ids[:, inputs.input_ids.shape[1]:]

    if new_ids.numel() == 0:
        answer = "Я не смог сгенерировать ответ на этот запрос."
    else:
        answer = tokenizer.batch_decode(new_ids, skip_special_tokens=True)[0].strip()

    history.append({"role": "assistant", "content": answer})
    print("\n")  # перенос строки после стрима
    return answer


def main():
    tokenizer, model = load_model()

    history = [
        {
            "role": "system",
            "content": (
                "Ты умный, дружелюбный ассистент. "
                "Отвечай по-русски, по делу и достаточно кратко."
            ),
        }
    ]

    print("Скоростной чат с Qwen. Напиши 'exit' или 'выход' для выхода.\n")

    while True:
        try:
            user_msg = input("Ты: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if user_msg.lower() in {"exit", "выход", "quit"}:
            break
        if not user_msg:
            continue

        # тут можно подстраивать агрессивность/скорость
        chat_once_stream(
            tokenizer,
            model,
            history,
            user_msg=user_msg,
            max_new_tokens=192,   # уменьшишь до 96/128 — будет ещё быстрее
            temperature=0.7,      # 0.5–0.8 адекватный диапазон рандома
            top_p=0.9,
        )

    print("Пока!")


if __name__ == "__main__":
    main()
