from modules.base_model import BaseModel
from modules.model_factory import ModelFactory
from transformers import TextIteratorStreamer, AutoProcessor, AutoModelForCausalLM, AutoTokenizer
from transformers.utils import check_min_version
from qwen_vl_utils import process_vision_info


@ModelFactory.register("qwen-vl")
class QwenVLModel(BaseModel):
    """
    Support QwenVLModel with transformers library    
    Tested with transformers==4.49.0    
    """

    def __init__(self, model_path, **kwargs) -> None:
        super().__init__()

        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-VL", trust_remote_code=True)
        self.core_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL", device_map="cuda", trust_remote_code=True,
                                                               torch_dtype="auto")

    def support_image(self):
        return True

    def support_video(self):
        return False

    def chatbot_to_messages(self, chatbot) -> list[str]:
        messages = []
        for user_msg, _ in chatbot:
            if isinstance(user_msg, (tuple, list)):  # query is media path                
                if BaseModel.is_image_file(user_msg[0]):
                    messages.append({"image":user_msg[0]})
                else:
                    raise ValueError(f"Unsupported file type: {user_msg[0]}")
            elif isinstance(user_msg, str):  # query is text                
                if user_msg:
                    messages.append({"text": user_msg})

        return messages

    def predict(self, chatbot: list[list[str | tuple]], params: dict):        
        if len(chatbot) == 0 or not chatbot[-1][0] or chatbot[-1][1]:  # Empty user input or non-empty reply
            yield chatbot
        else:            
            messages = self.chatbot_to_messages(chatbot)
            query = self.tokenizer.from_list_format(messages)
            inputs = self.tokenizer(query, return_tensors='pt')
            inputs = inputs.to("cuda")
            output = self.core_model.generate(**inputs)[0]
            response = self.tokenizer.decode(output[len(inputs.input_ids[0]):], skip_special_tokens=False)
            chatbot[-1][-1] += response            
            
            yield chatbot, ""

    @classmethod
    def description(cls) -> str:
        return "Qwen-VL"
