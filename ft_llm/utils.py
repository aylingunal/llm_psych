
''' helper functions for formatting data for seq or nl '''
import random
import datasets

def read_human(fname="/home/gunala/LLMDialGen/human_data/human_dat_reformatted.json"):
    # data processing
    import json
    with open(fname,'r') as inf:
        human_dat = json.load(inf)
    return human_dat

def format_dat(tokenizer):
    human_dat = read_human()
    all_seq_texts = []
    for game_id in human_dat.keys():
        seq_text = ""
        # order is reversed
        round_ids = sorted([int(x) for x in human_dat[game_id].keys()])
        # random sample of data histories
        round_ids_cap = random.randint(0,8)
        for round_id in range(round_ids_cap):
            upd_round_id = abs(len(round_ids) - round_id - 1)
            # using <s> as separator token
            seq_text += "P1: " + human_dat[game_id][str(round_id)]['p1'] + \
                        ", P2: " + human_dat[game_id][str(round_id)]['p2'] + '.<s>'
        all_seq_texts.append(seq_text)

    def encode_texts(items):
        return tokenizer(items["texts"], truncation=True, padding="max_length")

    data_df = {"texts":all_seq_texts}
    data_ds = datasets.Dataset.from_dict(data_df)
    data_ds = data_ds.map(encode_texts,batched=True,num_proc=4,)
    data_ds = data_ds.train_test_split(test_size=.2)

    return data_ds







