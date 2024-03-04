from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, AutoPeftModelForCausalLM


def main(options):
    device = "cuda"  # the device to load the model onto

    tokenizer = AutoTokenizer.from_pretrained(options.model_path)

    # Alternatively, use `model = AutoPeftModelForCausalLM.from_pretrained(options.lora_path, device_map=device)`
    model = AutoModelForCausalLM.from_pretrained(options.model_path, device_map=device)

    if options.lora_path:
        model = PeftModel.from_pretrained(model, model_id=options.lora_path)

    prompt = "Do NER: A brown teddy bear in an orange shirt and some people."
    messages = [
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    generated_ids = model.generate(
        input_ids = model_inputs.input_ids,
        max_new_tokens=512
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(response)


def parse_arguments():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str)
    parser.add_argument("--lora-path", type=str)
    options = parser.parse_args()
    return options


if __name__ == "__main__":
    options = parse_arguments()
    main(options)
