def safe_import(module_name, class_name):
    """Safely import a class from a module, printing errors instead of crashing"""
    try:
        module = __import__(module_name, fromlist=[class_name])
        return getattr(module, class_name)
    except ImportError as e:
        print(f"Warning: Failed to import {class_name} from {module_name}: {e}")
        return None
    except Exception as e:
        print(f"Warning: Error importing {class_name} from {module_name}: {e}")
        return None

# Attempt to import all models safely
safe_import('modules.auto_model', 'AutoModel')
safe_import('modules.qwen2_vl_model', 'Qwen2VLModel')
safe_import('modules.qwen_vl_model', 'QwenVLModel')
safe_import('modules.gguf_model', 'GGUFModel')
safe_import('modules.minicpm_v_model', 'MiniCPMModel')
safe_import('modules.phi_vision_model', 'PhiVisionModel')
safe_import('modules.llava_model', 'LlavaModel')
safe_import('modules.internvl_model', 'InternVL2Model')
safe_import('modules.video_llama', 'VideoLLaMA')
safe_import('modules.janus_model', 'JanusModel')
safe_import('modules.internvideo_model', 'InternVideoModel')
safe_import('modules.pta_model', 'PTAModel')