from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM

def main(model_path, lora_path, merged_model_path):
    # Set torch_dtype='auto' to use the dtype in the orginal model's config
    # The original model is often already in eval mode, but adding `.eval()` is a good style
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto", device_map='auto').eval()
    model = PeftModel.from_pretrained(model, model_id=lora_path)

    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(merged_model_path)

    tokenizer = AutoTokenizer.from_pretrained(lora_path, trust_remote_code=True)
    tokenizer.save_pretrained(merged_model_path)

def parse_arguments():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--lora_path', type=str)
    parser.add_argument('--merged_model_path', type=str)
    return parser.parse_args()

if __name__ == "__main__":
    options = parse_arguments()
    main(options.model_path, options.lora_path, options.merged_model_path)
