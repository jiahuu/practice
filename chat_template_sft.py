from transformers import AutoTokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")

def tokenizer_for_sft(tokenizer, conversation, max_length=512):
    assert conversation[-1]["role"] == "assistant"

    prompt = conversation[:-1]
    resp = conversation[-1]["content"]

    prompt_str = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)

    prompt_ids = tokenizer(prompt_str, add_special_tokens=False)["input_ids"]
    response_ids = tokenizer(resp + tokenizer.eos_token, apply_special_tokens=False)["input_ids"]

    input_ids = prompt_ids + response_ids
    # pytorch 的 F.cross_entropy 默认 ignore_index = -100, 即 labels = -100的位置不参与loss计算
    labels = [-100]*len(prompt_ids) + response_ids

    # padding
    pad_len = max_length - len(input_ids)
    input_ids = input_ids + [tokenizer.pad_token_id]*pad_len
    labels = labels + [-100]*pad_len

    return {
        "input_ids": torch.tensor(input_ids),
        "labels": torch.tensor(labels),
        "response_mask": torch.tensor([0]*len(prompt_ids) + [1]*len(response_ids) + [0]*pad_len)
    }

conversation = [
    {"role": "system", "content": "你是代码助手"},
    {"role": "user", "content": "写冒泡排序"},
    {"role":"assistant","content":"def bubble_sort(arr):\n    for i in range(len(arr)):\n        for j in range(len(arr)-i-1):\n            if arr[j]>arr[j+1]: arr[j],arr[j+1]=arr[j+1],arr[j]\n    return arr"}
]

output = tokenizer_for_sft(tokenizer, conversation)
print("response_mask中1的个数：", output["response_mask"].sum().item())
print("label中非 -100 的个数：", (output["labels"]!= -100 ).sum().item())

