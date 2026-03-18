#!/usr/bin/env python3
"""
风格改写脚本：调用 LLM API, 参考 train_data.json 中的语言风格，
将 replay_data.json 中的回答改写为目标风格，只做语言风格转换，不改动逻辑和内容。

所有配置从 .env 读取，参考 .env.example
"""

import json
import os
import random
import sys
import time
from pathlib import Path

try:
    from openai import OpenAI
except ImportError:
    print("错误：需要安装 openai 包 (pip install openai)")
    sys.exit(1)


def load_dotenv():
    for p in [Path(__file__).resolve().parent.parent / ".env", Path.cwd() / ".env"]:
        if p.is_file():
            with open(p, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, _, value = line.partition("=")
                    key = key.strip()
                    value = value.strip().strip("'\"")
                    if key and key not in os.environ:
                        os.environ[key] = value
            return


def build_rewrite_system_prompt(style_examples: list, persona: str) -> str:
    parts = [
        "你是一个语言风格转换助手。你的任务是将助手的回答改写成特定人物的说话风格，只做风格转换，不改变回答的逻辑和内容。",
    ]

    if persona:
        parts += ["", "## 角色人设", "", persona]

    parts += ["", "## 目标风格示例（学习对方的说话特点，不要复制内容）", ""]
    for i, conv in enumerate(style_examples):
        parts.append(f"示例 {i + 1}:")
        for msg in conv["conversations"]:
            label = "用户" if msg["role"] == "user" else "对方"
            parts.append(f"  {label}: {msg['content']}")
        parts.append("")

    parts += [
        "## 改写要求",
        "",
        "- 只改变语言风格，保留原回答的所有知识内容、逻辑和事实",
        "- 用目标人物的口吻表达，语气要自然、口语化",
        "- 如果原回答很长很正式，可以保留核心知识点，用更轻松的方式表达",
        "- 直接输出改写后的内容，不要加任何说明或解释",
    ]
    return "\n".join(parts)


def rewrite_answer(client, model, system_prompt, question, original_answer, temperature, max_retries=3) -> str:
    user_prompt = f"将下面的回答改写成目标风格：\n\n问题：{question}\n\n原回答：\n{original_answer}"

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                max_tokens=2048,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 2 ** attempt
                print(f"    API 错误: {e}, {wait}s 后重试...")
                time.sleep(wait)
            else:
                print(f"    API 错误: {e}, 跳过此条")
    return original_answer  # 失败时保留原始内容


def main():
    load_dotenv()

    api_key = os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("OPENAI_BASE_URL")
    model = os.environ.get("REWRITE_MODEL") or os.environ.get("AUGMENT_MODEL")
    for name, val in [("OPENAI_API_KEY", api_key), ("OPENAI_BASE_URL", base_url)]:
        if not val:
            print(f"错误：请在 .env 中配置 {name}")
            sys.exit(1)
    if not model:
        print("错误：请在 .env 中配置 REWRITE_MODEL 或 AUGMENT_MODEL")
        sys.exit(1)

    persona = os.environ.get("REWRITE_SYSTEM_PROMPT", "")
    style_ref_path = os.environ.get("REWRITE_STYLE_REF", "data/train_data.json")
    input_path = os.environ.get("REWRITE_INPUT", "data/replay_data.json")
    few_shot = int(os.environ.get("REWRITE_FEW_SHOT", "3"))
    temperature = float(os.environ.get("REWRITE_TEMPERATURE", "0.7"))

    p = Path(input_path)
    output_path = os.environ.get("REWRITE_OUTPUT") or str(p.parent / f"{p.stem}_styled{p.suffix}")

    # 加载风格参考数据
    if not Path(style_ref_path).is_file():
        print(f"错误：风格参考文件不存在: {style_ref_path}")
        sys.exit(1)
    with open(style_ref_path, encoding="utf-8") as f:
        style_data = json.load(f)
    print(f"已加载风格参考: {style_ref_path}({len(style_data)} 组对话)")

    # 加载待改写数据
    if not Path(input_path).is_file():
        print(f"错误：输入文件不存在: {input_path}")
        sys.exit(1)
    with open(input_path, encoding="utf-8") as f:
        replay_data = json.load(f)

    conversations = replay_data.get("conversations", [])
    print(f"已加载待改写数据: {input_path}({len(conversations)} 组对话)")

    if not persona:
        print("提示：未设置 REWRITE_SYSTEM_PROMPT, 建议在 .env 中配置角色人设以获得更好效果")

    actual_few_shot = min(few_shot, len(style_data))
    print(f"每条随机抽取 {actual_few_shot} 个风格示例")

    client = OpenAI(base_url=base_url, api_key=api_key)

    rewritten_conversations = []
    for i, conv in enumerate(conversations):
        msgs = conv.get("conversations", [])
        if len(msgs) < 2:
            rewritten_conversations.append(conv)
            continue

        # 提取 user 问题和 assistant 回答（每组只取第一对）
        question = next((m["content"] for m in msgs if m["role"] == "user"), "")
        original_answer = next((m["content"] for m in msgs if m["role"] == "assistant"), "")

        if not question or not original_answer:
            rewritten_conversations.append(conv)
            continue

        style_examples = random.sample(style_data, actual_few_shot)
        system_prompt = build_rewrite_system_prompt(style_examples, persona)
        print(f"  [{i + 1}/{len(conversations)}] 改写: {question[:30]}...")
        new_answer = rewrite_answer(client, model, system_prompt, question, original_answer, temperature)

        rewritten_conv = {
            "conversations": [
                {"role": "user", "content": question},
                {"role": "assistant", "content": new_answer},
            ]
        }
        rewritten_conversations.append(rewritten_conv)

    output_data = {
        "system_prompt": persona,
        "conversations": rewritten_conversations,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    print(f"\n已保存: {output_path} ({len(rewritten_conversations)} 组对话)")


if __name__ == "__main__":
    main()
