
LLAMA2_PROMPT = "<s>[INST] <<SYS>> {} <</SYS>> {} [/INST]"
SYSTEM_PROMPT = "Imagine that you and your criminal partner have been placed under arrest and are being interrogated in separate rooms. Both you and your partner have the option to either 'Silence' or 'Confess'. If both of you are silent, you both receive a prison sentence of 1 year. If one of you confesses and the other stays silent, the partner that confesses will receive a prison sentence of 0 years while the one of the stayed silent will receive a sentence of 5 years. If both of you confess, you will both receive sentences of 3 years. Your goal is to minimize your prison sentence. Respond in one word, either 'Silence' or 'Confess'."
USER_PROMPT = "You are Player {}, and this is the history of actions so far: {}. Will you choose 'Silence' or 'Confess' in the next round? Please answer in 1 word."

# for pytorch stuff
import torch
DEVICE = 'cpu' #'cuda' if torch.cuda.is_available() else 'cpu'

class Strategy():

    game_history = {}
    player_id = ''
    strategy = ''
    strategy_type = '' # ['hf','repl','classic','bandit']

    def __init__(self, strategy_in, strategy_type_in):
        self.strategy = strategy_in
        self.strategy_type = strategy_type_in
        if self.strategy_type == 'hf':
            self.init_hf_model()

    def next_action(self):
        if self.strategy_type == 'hf':
            return self.generate_hf()

    def add_to_hist(self, turn_num, self_play, other_play):
        self.game_history[turn_num] = {'self':self_play,'other':other_play}

    def format_hf_prompt(self):
        messages = [ { "role": "system","content": SYSTEM_PROMPT},
                     {"role": "user", "content": USER_PROMPT.format(self.player_id,self.game_history)}]
        start_prompt = "<s>[INST] "
        end_prompt = " [/INST]"
        conversation = []
        for index, message in enumerate(messages):
            if message["role"] == "system" and index == 0:
                conversation.append(f"<<SYS>>\n{message['content']}\n<</SYS>>\n\n")
            elif message["role"] == "user":
                conversation.append(message["content"].strip())
            else:
                conversation.append(f" [/INST] {message['content'].strip()} </s><s>[INST] ")
        return start_prompt + "".join(conversation) + end_prompt

    def init_hf_model(self):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import transformers
        import torch
        model = AutoModelForCausalLM.from_pretrained(self.strategy)
        tokenizer = AutoTokenizer.from_pretrained(self.strategy, device_map='auto')
        tokenizer.pad_token = tokenizer.eos_token
        # load to device
        model.to(DEVICE)
       # tokenizer.to(DEVICE)
        self.model = model
        self.tokenizer = tokenizer

    def generate_hf(self):
        prompt = self.format_hf_prompt()
        input_ids = self.tokenizer(prompt, padding=True, return_tensors="pt")
        input_ids['input_ids'] = input_ids['input_ids'].cpu()#.cuda()#.cpu()
        input_ids['attention_mask'] = input_ids['attention_mask'].cpu()#.cuda()#.cpu()
        num_input_tokens = input_ids['input_ids'].shape[1]
        outputs = self.model.generate(input_ids['input_ids'], attention_mask=input_ids['attention_mask'].half(),
                                    max_new_tokens=10, do_sample=True, pad_token_id=self.tokenizer.pad_token_id)
        generation = self.tokenizer.batch_decode(outputs[:, num_input_tokens:], skip_special_tokens=True)

        return generation
 