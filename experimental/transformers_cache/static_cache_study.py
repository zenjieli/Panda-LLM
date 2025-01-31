import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, StaticCache


def reset_cache(prompt_cache: StaticCache, pre_prompt_len: int) -> None:
    for k_cache in prompt_cache.key_cache:
        k_cache[:, :, pre_prompt_len:, :] = 0

    for v_cache in prompt_cache.value_cache:
        v_cache[:, :, pre_prompt_len:, :] = 0


def main():
    model_id = "Qwen/Qwen2.5-0.5B-Instruct"  # "meta-llama/Llama-3.2-1B-Instruct"
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    messages = [[{"role": "user", "content": "We know x=1 and y=2."}],
                [{"role": "user", "content": "We know x=1 and y=2. What is x plus y?"}],
                [{"role": "user", "content": "We know x=1 and y=2. What is x times y?"}]]
    prompt_inputs = tokenizer.apply_chat_template(
        messages[0], add_generation_prompt=True, return_tensors="pt", return_dict=True).to("cuda")

    prefix_len = len(prompt_inputs.input_ids[0]) - 5
    prompt_cache = StaticCache(config=model.config, max_batch_size=1, max_cache_len=prefix_len +
                            256, device="cuda", dtype=model.config.torch_dtype)

    cache_position = torch.arange(prompt_inputs.input_ids.shape[1], dtype=torch.int64, device="cuda")

    # Run forward without grad to be abel to copy
    with torch.no_grad():
        outputs = model(**prompt_inputs, cache_position=cache_position, past_key_values=prompt_cache, use_cache=True)

    conversations = [tokenizer.apply_chat_template(messages[1], add_generation_prompt=True, return_tensors="pt", return_dict=True),
            tokenizer.apply_chat_template(messages[2], add_generation_prompt=True, return_tensors="pt", return_dict=True)]

    max_new_tokens = 100
    for inputs in conversations:
        inputs = inputs.to("cuda")
        reset_cache(prompt_cache, prefix_len)

        print(f"Input: {tokenizer.decode(inputs.input_ids[0])}")
        responses = []

        outputs = model.generate(**inputs,
                                past_key_values=prompt_cache,
                                max_new_tokens=max_new_tokens,
                                do_sample=False,
                                return_dict_in_generate=True)
        response = tokenizer.batch_decode(outputs["sequences"])[0]
        responses.append(response)

        print(responses)

if __name__ == "__main__":
    main()
