import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, Qwen2Config

model_id = "Qwen/Qwen2.5-1.5B-Instruct"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side='left')
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

config = AutoConfig.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(model_id, device_map='auto', torch_dtype="auto")
model.eval()

# Count the number of trainable parameters
accum = 0
for name, p in model.named_parameters():
    accum += p.numel()
    print(f"{name}: {list(p.shape)} {p.numel()}\tAccumulated: {accum}")

from llama_parameters import compute_total_parameters
assert compute_total_parameters(config, True) == 1777088000

messages = [[{"role": "user", "content": "2+3=? Answer with a single number."}],
    [{"role": "user", "content": "Hi"}]]
model_inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors='pt', padding=True).to(device)
generate_kwargs = {'input_ids': model_inputs,
                   'do_sample': False}
outputs = model(**generate_kwargs)

token_ids = outputs.logits[:,-1:].argmax(dim=-1)
print(tokenizer.batch_decode(token_ids, skip_special_tokens=True))


