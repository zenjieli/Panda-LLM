import torch
import copy

def get_raw_input(model_tokenizer, seq_length, inputs, device='cuda'):
    model_input_ids = []
    model_attention_mask = []
    model_token_type_ids = []
    model_position_ids = []
    inputs_aux = copy.deepcopy(inputs)

    for key in inputs.keys():
        if key == "input_ids":
            for input_ids in inputs["input_ids"]:
                input_ids = input_ids.tolist()
                pad_token = model_tokenizer.pad_token_id if model_tokenizer.pad_token_id else 0
                input_ids = ([pad_token] * seq_length) + input_ids
                model_input_ids.append(input_ids)

        elif key == "attention_mask":
            for attention_mask in inputs['attention_mask']:
                attention_mask = attention_mask.tolist()
                attention_mask = ([0] * seq_length) + attention_mask
                model_attention_mask.append(attention_mask)
        elif key == "token_type_ids":
            pad_token_segment_id = 0
            for token_type_ids in inputs['token_type_ids']:
                token_type_ids = token_type_ids.tolist()
                token_type_ids = token_type_ids + ([pad_token_segment_id] * seq_length)
                model_token_type_ids.append(token_type_ids)
        '''
        elif key == "position_ids":
            position_ids = inputs['position_ids']
            for i in range(origin_length, max_length):
                position_ids.append(i)
            model_position_ids.append(position_ids)
        '''

    if len(model_input_ids) > 0:
        inputs_aux.update({"input_ids": torch.tensor(model_input_ids).to(device)})
    if len(model_attention_mask) > 0:
        inputs_aux.update({"attention_mask": torch.tensor(model_attention_mask).to(device)})
    if len(model_token_type_ids) > 0:
        inputs_aux.update({'token_type_ids': torch.tensor(model_token_type_ids).to(device)})

    '''
    if len(model_position_ids) > 0:
        inputs.update({'position_ids': torch.tensor(model_position_ids).to(device)})
    '''

    return inputs_aux

def get_memory():
    world_size = torch.cuda.device_count()
    memory = 0
    for i in range(world_size):
        device = 'cuda:' + str(i)
        memory += torch.cuda.max_memory_allocated(device=device)
        torch.cuda.reset_peak_memory_stats(device=device)
    
    return memory    