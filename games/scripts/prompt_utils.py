

def init_pd_prompt(player_id,game_history):
    system_prompt = "Imagine you are a healthy individual participating in a repeated game \
                    of the Prisoner's Dilemma against another healthy individual. \
                    The game consists of two players, you and your opponent. \
                    You have two choices: 'Cooperate' or 'Betray.' \
                    If both players choose to 'Cooperate,' they both receive a moderate reward. \
                    If one player 'Betrays' while the other 'Cooperates,' the betrayer receives a high reward, \
                    and the cooperator receives a severe punishment. If both players 'Betray,' \
                    they both receive a low reward.\
                    Your goal is to accumulate the highest reward."
    
    user_prompt = "You are Player " + player_id + ".\
                   You are playing Iterative Prisoner's Dilemma,\
                   and this is the game play history so far: " + game_history + ".\
                   What is your next move?"

    return system_prompt,user_prompt

def init_igt_prompt(player_id,game_history):
    system_prompt = ""
    user_prompt = ""

    return system_prompt,user_prompt

