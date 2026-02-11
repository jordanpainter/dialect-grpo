def build_chat_prompt(tokenizer, system_prompt: str, user_prompt: str) -> str:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})

    if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # Fallback for TinyLlama-style tags
    if system_prompt:
        return f"<|system|>\n{system_prompt}\n<|user|>\n{user_prompt}\n<|assistant|>\n"
    return f"<|user|>\n{user_prompt}\n<|assistant|>\n"
