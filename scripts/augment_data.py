#!/usr/bin/env python3
"""
数据增强脚本：调用 LLM API 从 few-shot 示例中学习风格，批量生成新对话
兼容所有 OpenAI 格式 API（DeepSeek、OpenAI、本地模型等）

配置通过 .env 文件指定，参考 .env.example
"""

import json
import os
import random
import re
import sys
import time
from pathlib import Path

try:
    from openai import OpenAI
except ImportError:
    print("错误：需要安装 openai 包（pip install openai）")
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


TOPIC_POOL = [
    "约吃饭/聚餐", "天气变化", "工作或学习上的吐槽", "网购/快递到了",
    "打游戏/开黑", "追剧/综艺推荐", "旅行/出游计划", "运动/健身",
    "节日祝福/节日安排", "日常吐槽/抱怨", "借东西/帮忙", "早安/晚安闲聊",
    "生病/身体不舒服", "周末计划", "搬家/租房", "拍照/修图/发朋友圈",
    "宠物", "做饭/点外卖", "考试/作业/论文", "约看电影",
    "发红包/转账", "推荐歌曲/音乐", "讨论某个朋友/共同认识的人",
    "吐槽某件尴尬的事", "问路/问地址", "约逛街/购物", "讨论发型/穿搭",
    "讨论某个新闻/热搜", "请教问题/求推荐", "分享搞笑的事/段子",
    "讨论放假安排", "约打球/跑步", "回忆以前的事", "送礼物/选礼物",
    "讨论某家店/某个地方", "通知/传达消息", "还钱/AA制",
    "迟到/爽约道歉", "安慰/鼓励对方", "讨论毕业/升学/找工作",
    "约KTV/唱歌", "分享照片/视频", "讨论搞笑视频/表情包",
    "约自习/一起学习", "讨论某个APP/软件", "问对方在干嘛",
    "讨论家人/家里的事", "聊睡眠/熬夜", "讨论某个老师/同事",
]


def load_data(path: str) -> list:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        print("错误：输入文件应为 JSON 数组")
        sys.exit(1)
    print(f"已加载 {len(data)} 组对话")
    return data


def build_system_prompt(examples: list, persona: str = None) -> str:
    parts = ["你是一个对话数据生成器。仔细观察以下示例的说话风格，生成风格一致的微信日常聊天对话。"]

    if persona:
        parts += ["", "## 角色人设", "", persona]

    parts += ["", "## 风格示例（学习说话习惯，不要复制内容）", ""]
    for i, conv in enumerate(examples):
        parts.append(f"示例 {i + 1}:")
        for msg in conv["conversations"]:
            label = "用户" if msg["role"] == "user" else "对方"
            parts.append(f"  {label}: {msg['content']}")
        parts.append("")

    parts += [
        "## 输出格式",
        "",
        '严格输出 JSON 数组，每项包含 "conversations" 字段，消息交替 user/assistant，以 user 开头、assistant 结尾，每组 4-10 条。',
        '连发消息用 \\n 分隔（如："到啦\\n你们呢"）。只输出 JSON，不要其他内容。',
    ]
    return "\n".join(parts)


def parse_json_response(text: str) -> list:
    text = text.strip()
    for candidate in [
        text,
        re.sub(r'^```(?:json)?\s*\n?', '', re.sub(r'\n?```\s*$', '', text, flags=re.MULTILINE), flags=re.MULTILINE).strip(),
    ]:
        try:
            result = json.loads(candidate)
            if isinstance(result, list):
                return result
            if isinstance(result, dict):
                for key in ("data", "conversations", "dialogues", "result"):
                    if key in result and isinstance(result[key], list):
                        return result[key]
        except json.JSONDecodeError:
            pass

    match = re.search(r'\[[\s\S]*\]', text)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return []


def generate_batch(client, model, system_prompt, topics, batch_size, temperature, max_retries=3):
    topic_list = "\n".join(f"  {i + 1}. {t}" for i, t in enumerate(topics))
    user_prompt = f"请生成 {batch_size} 组微信聊天对话，话题分别是：\n{topic_list}\n\n每组 4-10 条消息，严格按 JSON 格式输出。"

    for attempt in range(max_retries):
        try:
            kwargs = dict(
                model=model,
                messages=[{"role": "system", "content": system_prompt},
                          {"role": "user", "content": user_prompt}],
                temperature=temperature,
                max_tokens=4096,
            )
            if attempt == 0:
                try:
                    response = client.chat.completions.create(**kwargs, response_format={"type": "json_object"})
                except Exception:
                    response = client.chat.completions.create(**kwargs)
            else:
                response = client.chat.completions.create(**kwargs)

            result = parse_json_response(response.choices[0].message.content)
            if result:
                return result
            if attempt < max_retries - 1:
                print("    JSON 解析失败，重试中...")
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 2 ** attempt
                print(f"    API 错误: {e}，{wait}s 后重试...")
                time.sleep(wait)
            else:
                print(f"    API 错误: {e}，跳过此批次")
    return []


def validate_and_fix(conversations: list) -> list:
    valid = []
    for conv in conversations:
        if not isinstance(conv, dict) or "conversations" not in conv:
            continue
        msgs = conv["conversations"]
        if not isinstance(msgs, list) or len(msgs) < 2:
            continue
        if not all(isinstance(m, dict) and m.get("role") in ("user", "assistant")
                   and isinstance(m.get("content"), str) and m["content"].strip()
                   for m in msgs):
            continue

        if msgs[0]["role"] != "user":
            msgs = msgs[1:]

        # 确保角色交替
        fixed = [msgs[0]]
        for msg in msgs[1:]:
            if msg["role"] != fixed[-1]["role"]:
                fixed.append(msg)

        if fixed[-1]["role"] == "user":
            fixed = fixed[:-1]
        if len(fixed) >= 2:
            valid.append({"conversations": fixed})
    return valid


def main():
    load_dotenv()

    api_key = os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("OPENAI_BASE_URL")
    model = os.environ.get("AUGMENT_MODEL")
    for name, val in [("OPENAI_API_KEY", api_key), ("OPENAI_BASE_URL", base_url), ("AUGMENT_MODEL", model)]:
        if not val:
            print(f"错误：请在 .env 中配置 {name}")
            sys.exit(1)

    input_path = os.environ.get("AUGMENT_INPUT", "data/train_data.json")
    num = int(os.environ.get("AUGMENT_NUM", "20"))
    batch_size = int(os.environ.get("AUGMENT_BATCH_SIZE", "5"))
    temperature = float(os.environ.get("AUGMENT_TEMPERATURE", "0.9"))
    few_shot = int(os.environ.get("AUGMENT_FEW_SHOT", "6"))

    p = Path(input_path)
    output_path = os.environ.get("AUGMENT_OUTPUT") or str(p.parent / f"{p.stem}_augmented{p.suffix}")

    # 加载人设
    persona = None
    for cp in ["data/config.json", str(Path(__file__).resolve().parent.parent / "data" / "config.json")]:
        if Path(cp).is_file():
            persona = json.load(open(cp, encoding="utf-8")).get("system_prompt", "")
            if persona:
                print(f"已加载人设: {cp}")
            break

    data = load_data(input_path)
    examples = random.sample(data, min(few_shot, len(data)))
    system_prompt = build_system_prompt(examples, persona)
    print(f"使用 {len(examples)} 个 few-shot 示例，system prompt {len(system_prompt)} 字符")

    client = OpenAI(base_url=base_url, api_key=api_key)
    topics = random.sample(TOPIC_POOL, len(TOPIC_POOL))

    all_generated = []
    total_batches = (num + batch_size - 1) // batch_size
    for batch_idx in range(1, total_batches + 1):
        current = min(batch_size, num - len(all_generated))
        if current <= 0:
            break
        start = ((batch_idx - 1) * batch_size) % len(topics)
        batch_topics = [topics[(start + j) % len(topics)] for j in range(current)]

        print(f"  [{batch_idx}/{total_batches}] 生成 {current} 个对话 (已完成 {len(all_generated)}/{num})...")
        validated = validate_and_fix(generate_batch(client, model, system_prompt, batch_topics, current, temperature))
        all_generated.extend(validated)
        print(f"    成功 {len(validated)} 个" if validated else "    本批次无有效对话")

    print(f"\n共生成 {len(all_generated)} 个有效对话（目标 {num}）")
    if not all_generated:
        print("错误：未生成任何有效对话，请检查 API 配置")
        sys.exit(1)

    output_data = data + all_generated
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    print(f"已保存: {output_path}（原始 {len(data)} + 生成 {len(all_generated)} = {len(output_data)} 组）")


if __name__ == "__main__":
    main()
