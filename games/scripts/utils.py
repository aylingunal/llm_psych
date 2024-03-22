# llama stuff
SYSTEM_PROMPT = "Imagine that you and your criminal partner have been placed under arrest and are being interrogated in separate rooms. Both you and your partner have the option to either 'Silence' or 'Confess'. If both of you are silent, you both receive a prison sentence of 1 year. If one of you confesses and the other stays silent, the partner that confesses will receive a prison sentence of 0 years while the one of the stayed silent will receive a sentence of 5 years. If both of you confess, you will both receive sentences of 3 years. Your goal is to minimize your prison sentence. Respond in one word, either 'Silence' or 'Confess'."
USER_PROMPT = "You are Player {}, and this is the history of actions so far: {}. Will you choose 'Silence' or 'Confess' in the next round? Please answer in 1 word."

# for pytorch stuff
import torch
from transformers import AutoTokenizer, \
                         AutoModelForCausalLM, \
                         LlamaTokenizer, \
                         LlamaForCausalLM, \
                         BitsAndBytesConfig
import transformers
from bandit_utils import *

import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1,2,3"

# from huggingface_hub import login
# login()
os.environ['HF_TOKEN'] = ""
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu' # need cuda for peft

class PDModel():
    game_history = {}
    player_id = ""
    model_type = "" # ['hf','repl','classic','bandit']
    model_name = ""
    lm_prompt = "" # if model is lm, this is the prompt we use 

    ### INIT ###
    def __init__(self, 
                 model_name_in,
                 player_id_in,
                 cooperate_token_in="Silence",
                 defect_token_in="Confess"):
        self.model_name = model_name_in
        self.player_id = player_id_in

        self.cooperate_token = cooperate_token_in
        self.defect_token = defect_token_in

        # initialize hf models
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        # if player_id_in == "1":
        #     self.device = "cuda:0"
        # if player_id_in == "2":
        #     self.device = "cuda:1"
        if self.model_name == "google/gemma-7b":
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name,
                                                             # quantization_config=bnb_config,
                                                             # device_map="",#.to(self.device)
                                                              )
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.model_name == "aegunal/FT_IPD_gemma7b":
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name,
                                                              quantization_config=bnb_config)#.to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained("google/gemma-7b")
        if self.model_name == "mistralai/Mistral-7B-v0.1":
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name,
                                                              quantization_config=bnb_config)#.to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.model_name == "aegunal/FT_IPD_mistral7b":
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name,
                                                              quantization_config=bnb_config)#.to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
        if self.model_name == "meta-llama/Llama-2-7b-hf":
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name,
                                                              quantization_config=bnb_config)#.to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.model.config.eos_token_id
        if self.model_name == "aegunal/FT_IPD_llama7b":
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name,
                                                              quantization_config=bnb_config)#.to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.model.config.eos_token_id
        if self.model_name == "tiiuae/falcon-7b":
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name,
                                                        quantization_config=bnb_config)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.model.config.eos_token_id

        # initialize classic strategies
        if self.model_name == "gt":
            pass
        if self.model_name == "t4t":
            pass
        if self.model_name == "cooperate":
            pass 
        if self.model_name == "defect":
            pass
        if self.model_name == "random":
            pass

    ### GENERATE ACTION ###
    
    # generate action for classic strategies
    def next_action_classic(self):
        if self.model_name == "cooperate":
            return self.cooperate_token
        if self.model_name == "defect":
            return self.defect_token
        if self.model_name == "gt":
            # if not first turn 
            if len(self.game_history.keys()) > 0:
                num_turns = sorted(self.game_history.keys(),
                                   key=lambda x: int(x),
                                   reverse=True)
                last_turn = num_turns[len(num_turns)]
                if self.game_history[last_turn]["self"] == self.defect_token:
                    return self.defect_token
                elif self.game_history[last_turn]["other"] == self.defect_token:
                    return self.defect_token
                else:
                    return self.cooperate_token
            else:
                return self.cooperate_token
        if self.model_name == "t4t":
            # if not first turn
            if len(self.game_history.keys()) > 0:
                num_turns = sorted(self.game_history.keys(),
                                   key=lambda x: int(x),
                                   reverse=True)
                last_turn = num_turns[len(num_turns)]
                # if oppponent last defected, i also defect
                if self.game_history[last_turn]["other"] == self.defect_token:
                    return self.defect_token
                # in all other cases, can cooperate
                return self.cooperate_token

    def next_action_hf(self):
        # format main prompt
        prompt = SYSTEM_PROMPT + USER_PROMPT.format(self.player_id,self.game_history)
        input_ids = self.tokenizer([prompt], 
                                   return_tensors="pt",
                                   max_length=4096)#.to(self.device)
        # generate; this should work for all models
        with torch.no_grad():
            outputs = self.model.generate(**input_ids,
                                          output_scores=True,
                                          output_logits=True,
                                          return_dict_in_generate=True,
                                          max_new_tokens=1,
                                          temperature=0.75,
                                          )
        # get idx of target toks in vocab then map to corresponding scores from last batch
        coop_tok_id = self.tokenizer.encode(self.cooperate_token.lower(),return_tensors="pt")[0][1]
        def_tok_id = self.tokenizer.encode(self.defect_token.lower(),return_tensors="pt")[0][1]
        batch_size = len(outputs.scores)
        assert batch_size == 1, "batch size not 1"
        coop_score = outputs.scores[batch_size-1][0][coop_tok_id]
        def_score = outputs.scores[batch_size-1][0][def_tok_id]
        action = "Silence" if coop_score > def_score else "Confess"

        # format first-level theory of mind prompt
        upd_user_prompt = "You are Player {}, and this is the history of actions so far: {}. Do you think your opponent will play 'Silence' or 'Confess'?."
        prompt = SYSTEM_PROMPT + upd_user_prompt.format(self.player_id,self.game_history)
        input_ids = self.tokenizer([prompt], 
                                   return_tensors="pt",
                                   max_length=4096)#.to(self.device)
        # generate; this should work for all models
        with torch.no_grad():
            outputs = self.model.generate(**input_ids,
                                          output_scores=True,
                                          output_logits=True,
                                          return_dict_in_generate=True,
                                          max_new_tokens=1,
                                          temperature=0.75,
                                          )
        # get idx of target toks in vocab then map to corresponding scores from last batch
        coop_tok_id = self.tokenizer.encode(self.cooperate_token.lower(),return_tensors="pt")[0][1]
        def_tok_id = self.tokenizer.encode(self.defect_token.lower(),return_tensors="pt")[0][1]
        batch_size = len(outputs.scores)
        assert batch_size == 1, "batch size not 1"
        first_lvl_coop_score = outputs.scores[batch_size-1][0][coop_tok_id]
        first_lvl_def_score = outputs.scores[batch_size-1][0][def_tok_id]
        first_lvl_action = "Silence" if coop_score > def_score else "Confess"

        # record round info
        round_info_dict = {'action':action,
                           'coop_score':coop_score,
                           'def_score':def_score,
                           'first_lvl_action':first_lvl_action,
                           'first_lvl_coop_score':first_lvl_coop_score,
                           'first_lvl_def_score':first_lvl_def_score}
        return round_info_dict

    def generate_mab(self):
        generation = 'Silence' if self.model.choose_action() == 0 else 'Confess'
        return generation

    ### MISC ###

    def add_to_hist(self, turn_num, self_play, other_play):
        self.game_history[turn_num] = {'self':self_play,'other':other_play}
        # update bandit strategy if nec
        if self.model_name == 'mab':
            reward = self.model.get_reward(self_play,other_play)
            self.model.update_rewards((0 if self_play=='Silence' else 1),reward)

def print_all_models():
    all_models = ["google/gemma-7b",
                  "aegunal/FT_IPD_gemma7b",
                  "mistralai/Mistral-7B-v0.1",
                  "aegunal/FT_IPD_mistral7b",
                  "meta-llama/Llama-2-7b-hf",
                  "aegunal/FT_IPD_llama7b",
                  "tiiuae/falcon-7b"]
    
    for model_name in all_models:
        print(model_name)


#     def format_llama_prompt(self):
#         messages = [ { "role": "system","content": SYSTEM_PROMPT},
#                      {"role": "user", "content": USER_PROMPT.format(self.player_id,self.game_history)}]
#         start_prompt = "<s>[INST] "
#         end_prompt = " [/INST]"
#         conversation = []
#         for index, message in enumerate(messages):
#             if message["role"] == "system" and index == 0:
#                 conversation.append(f"<<SYS>>\n{message['content']}\n<</SYS>>\n\n")
#             elif message["role"] == "user":
#                 conversation.append(message["content"].strip())
#             else:
#                 conversation.append(f" [/INST] {message['content'].strip()} </s><s>[INST] ")
#         return start_prompt + "".join(conversation) + end_prompt


# print('loading test 1')
# test_model_1 = PDModel(model_name_in="google/gemma-7b",
#                        player_id_in="1",)
# print('loading test 2')
# test_model_2 = PDModel(model_name_in="google/gemma-7b",
#                        player_id_in="2",)

# print(test_model_1.next_action_hf())
# print(test_model_2.next_action_hf())




