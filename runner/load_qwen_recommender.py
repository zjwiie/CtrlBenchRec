import torch

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    GenerationConfig,
    logging as hf_logging,
    StoppingCriteria,
    StoppingCriteriaList
)

def _load_model(model_path: str):
    """优化模型加载：启用梯度检查点，优化padding处理"""
    # logger.info(f"🤖 加载模型: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, padding_side="left"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # 确保pad_token_id与eos_token_id一致
    tokenizer.pad_token_id = tokenizer.eos_token_id

    model_kwargs = {
        "trust_remote_code": True,
        # 1. 消除警告：将 torch_dtype 改为 dtype
        "dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32,

        # 2. 修复核心报错：将 device_map 改为 auto，让它自动识别你唯一的一张 4070
        "device_map": "auto",

        # 3. 修复核心报错：删掉不存在的 "1" 号显卡配置。
        # 对于 2B 模型，8GB 显存绰绰有余，不需要手动限制 max_memory。
        # 如果非要限制，只写 0 即可，例如 {"0": "7GB"}
    }

    # 修复后加载
    model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)

    # 启用梯度检查点（降低显存占用）
    # model.gradient_checkpointing_enable()
    # logger.info("✅ 启用梯度检查点优化显存占用")

    model.eval()
    return tokenizer, model


def inference(text, tokenizer, model):
    # 1. 准备输入
    # add_generation_prompt=True 会自动添加 Qwen 的对话模板（如果是 Chat 版模型）
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    # 2. 生成配置
    # max_new_tokens: 生成多少个字
    # temperature: 随机性（越高越有创意，越低越严谨）
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id
    )

    # 3. 解码输出
    # skip_special_tokens=True 会过滤掉 <|endoftext|> 等特殊标记
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # 如果你只想看回复部分（过滤掉输入的 Prompt），可以切片：
    # response = tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)

    return response


# --- 主程序运行 ---
if __name__ == "__main__":
    tokenizer, model = _load_model("../model/Qwen3___5-2B")

    prompt = "你是谁。"
    print(f"User: {prompt}")

    result = inference(prompt, tokenizer, model)
    print(f"Assistant: {result}")
