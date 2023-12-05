from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch

import os
os.environ['CUDA_VISIBLE_DEVICES'] = ""

LLAMA2_PROMPT = " <s>[INST] <<SYS>> \
                {  } \
                <</SYS>> \
                {  } [/INST] "

def load_model(model_name):
    # load model 1
    # device_map = 'balanced' when cpu only, 'auto' when gpu and cpu avail
    model_path = "/home/gunala/LLMDialGen/" + model_name
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float32, device_map="auto").eval()
    use_fast_tokenizer = "LlamaForCausalLM" not in model.config.architectures
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side='left', use_fast=use_fast_tokenizer)
    tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer

# generate the llama response provided   
def generate_llama_response(model, tokenizer, prompt):
    # generate individual response; assuming system prompt is staying the same
    input_ids = tokenizer(prompt, padding=True, return_tensors="pt")
    input_ids['input_ids'] = input_ids['input_ids'].cuda()#.cpu()
    input_ids['attention_mask'] = input_ids['attention_mask'].cuda()#.cpu()
    num_input_tokens = input_ids['input_ids'].shape[1]
    outputs = model.generate(input_ids['input_ids'], attention_mask=input_ids['attention_mask'].half(),
                                max_new_tokens=64, do_sample=True, pad_token_id=tokenizer.pad_token_id)
    generation = tokenizer.batch_decode(outputs[:, num_input_tokens:], skip_special_tokens=True)

    return generation