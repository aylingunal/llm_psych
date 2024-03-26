

BNBEnv 
- has cuda 12.11 torch
- has bitsandbytes-cuda116 AND bitsandbytes

QLORAFTEnv
- has cuda 12.11 torch
- works for fting llama7b, mistral7b, gemma7b for 2 GPUs; falcon7b runs out of mem on 3 GPUs; gpt2 runs into some problem w labels while training (issue w/ datacollator?)

