import argparse
from huggingface_hub import snapshot_download
import os

os.environ['CUDA_VISIBLE_DEVICES'] = ""
model_name = "meta-llama/Llama-2-7b-hf"
# options: 
# meta-llama/Llama-2-7b-hf
# meta-llama/Llama-2-13b-hf
# meta-llama/Llama-2-70b-hf

def main():    
    # ========== download model ========== #
    model_local_fpath = "./llama_model_7b"
    if not os.path.exists(model_local_fpath):
        print('downloading llama model')
        snapshot_download(repo_id=model_name, local_dir=model_local_fpath, token=True)
        print('download complete')
    else:
        print('model previously downloaded')

if __name__ == "__main__":
    main()