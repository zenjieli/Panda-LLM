from transformers import AutoModelForCausalLM, AutoTokenizer


def main(options):
    device = "cuda"  # the device to load the model onto

    model = AutoModelForCausalLM.from_pretrained(options.model_path, device_map=device)
    tokenizer = AutoTokenizer.from_pretrained(options.model_path)

    prompt = "Once upon a time"
    model_inputs = tokenizer([prompt], return_tensors="pt").to(device)

    generated_ids = model.generate(
        model_inputs.input_ids,
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
    options = parser.parse_args()
    return options


if __name__ == "__main__":
    options = parse_arguments()
    main(options)
