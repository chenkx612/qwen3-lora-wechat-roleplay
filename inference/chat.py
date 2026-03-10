#!/usr/bin/env python3
"""
本地推理脚本：使用微调后的模型进行交互式聊天
支持两种推理方式：
1. transformers + LoRA（默认）
2. llama.cpp（需要GGUF格式模型）
"""

import argparse
import sys
from pathlib import Path


def load_transformers_model(model_path: str, lora_path: str = None):
    """加载transformers模型（可选LoRA）"""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"加载模型: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )

    # 设置pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float32,  # CPU使用float32
        device_map="cpu",
        trust_remote_code=True
    )

    # 加载LoRA权重
    if lora_path and Path(lora_path).exists():
        print(f"加载LoRA权重: {lora_path}")
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, lora_path)
        model = model.merge_and_unload()  # 合并权重以加速推理

    model.eval()
    return model, tokenizer


def load_llama_cpp_model(model_path: str):
    """加载llama.cpp模型（GGUF格式）"""
    from llama_cpp import Llama

    print(f"加载GGUF模型: {model_path}")

    model = Llama(
        model_path=model_path,
        n_ctx=2048,
        n_threads=4,
        verbose=False
    )

    return model


def generate_transformers(model, tokenizer, messages: list, max_new_tokens: int = 256):
    """使用transformers生成回复"""
    import torch

    # 构建对话prompt（Qwen3格式，禁用thinking避免输出<think>标签）
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )

    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    # 解码生成的部分
    generated = outputs[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(generated, skip_special_tokens=True)

    return response.strip()


def generate_llama_cpp(model, messages: list, max_tokens: int = 256):
    """使用llama.cpp生成回复"""
    # 构建对话格式
    formatted_messages = []
    for msg in messages:
        formatted_messages.append({
            "role": msg["role"],
            "content": msg["content"]
        })

    response = model.create_chat_completion(
        messages=formatted_messages,
        max_tokens=max_tokens,
        temperature=0.7,
        top_p=0.9,
        repeat_penalty=1.1
    )

    return response["choices"][0]["message"]["content"].strip()


def chat_loop(model, tokenizer, backend: str, system_prompt: str = None):
    """交互式聊天循环"""
    print("\n" + "="*50)
    print("聊天已启动！输入 /quit 或 /exit 退出")
    print("输入 /clear 清空对话历史")
    print("="*50 + "\n")

    messages = []

    # 添加系统提示（可选）
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    while True:
        try:
            user_input = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break

        if not user_input:
            continue

        # 命令处理
        if user_input.lower() in ["/quit", "/exit"]:
            print("再见！")
            break

        if user_input.lower() == "/clear":
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            print("[对话历史已清空]")
            continue

        # 添加用户消息
        messages.append({"role": "user", "content": user_input})

        # 生成回复
        try:
            if backend == "transformers":
                response = generate_transformers(model, tokenizer, messages)
            else:
                response = generate_llama_cpp(model, messages)

            print(f"{response}\n")

            # 添加助手回复到历史
            messages.append({"role": "assistant", "content": response})

        except Exception as e:
            print(f"[生成错误: {e}]")
            # 移除失败的用户消息
            messages.pop()


def main():
    parser = argparse.ArgumentParser(description="本地聊天推理")
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="Qwen/Qwen3-0.6B",
        help="模型路径或HuggingFace模型ID"
    )
    parser.add_argument(
        "--lora", "-l",
        type=str,
        default=None,
        help="LoRA权重路径（仅transformers后端）"
    )
    parser.add_argument(
        "--backend", "-b",
        type=str,
        choices=["transformers", "llama.cpp"],
        default="transformers",
        help="推理后端"
    )
    parser.add_argument(
        "--system-prompt", "-s",
        type=str,
        default=None,
        help="系统提示词"
    )
    parser.add_argument(
        "--gguf",
        type=str,
        default=None,
        help="GGUF模型文件路径（llama.cpp后端）"
    )

    args = parser.parse_args()

    # 根据后端加载模型
    if args.backend == "transformers":
        model, tokenizer = load_transformers_model(args.model, args.lora)
        chat_loop(model, tokenizer, args.backend, args.system_prompt)
    else:
        if not args.gguf:
            print("错误：llama.cpp后端需要指定--gguf参数")
            sys.exit(1)
        model = load_llama_cpp_model(args.gguf)
        chat_loop(model, None, args.backend, args.system_prompt)


if __name__ == "__main__":
    main()
