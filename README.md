
dev notes
- torch version in DialGenEnv is cpu only
- llama input prompts include ['assistant','user','system']
    - **'assistant': i still don't really get this one; supposed to represent LLM response
    so i'm guessing it's more for fine-tuning or providing history of chat as input** --> [[i THINK this is only for openai models; llama only user and system]]
    - 'user': what the user is prompting the LLM
    - 'system': internal guiding of *how* the LLM should respond (e.g. 'pretend you are shakespeare')

code notes
- ignore llama/; this is an old folder and i'll probably delete it
- use llama_download.py script to download the model folder that will
  then be used for loading (i.e. from_pretrained({foldername}))