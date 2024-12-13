import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "meta-llama/Llama-3.2-1B-Instruct"
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype="auto")
model.eval()
messages = [[{"role": "user", "content": "2+3=? Answer with a single number."}],
            [{"role": "user", "content": "Hi"}]]
print(tokenizer.apply_chat_template(messages, add_generation_prompt=True, padding=True, tokenize=False))
model_inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, padding=True, return_tensors="pt").to(device)
generate_kwargs = {"input_ids": model_inputs,
                   "do_sample": False}
outputs = model(**generate_kwargs)
token_ids = outputs.logits[:,-1:].argmax(dim=-1)
generated_text = tokenizer.batch_decode(token_ids, skip_special_tokens=True)
print(generated_text)