import os.path as osp
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM

new_model_directory = osp.expanduser('~/models/lora/Qwen1.5-1.8B-Chat-Lora')
path_to_adapter_dir = osp.expanduser('~/workspace/github/Qwen1.5/examples/sft/output_qwen')
original_model_dir = 'models/Qwen1.5-1.8B-Chat'

# Set torch_dtype='auto' to use the dtype in the orginal model's config
# The original model is often already in eval mode, but adding `.eval()` is a good style
model = AutoModelForCausalLM.from_pretrained(original_model_dir, torch_dtype="auto", device_map='auto').eval()
model = PeftModel.from_pretrained(model, model_id=path_to_adapter_dir)

merged_model = model.merge_and_unload()
merged_model.save_pretrained(new_model_directory)

tokenizer = AutoTokenizer.from_pretrained(path_to_adapter_dir, trust_remote_code=True)
tokenizer.save_pretrained(new_model_directory)
