import random

class MAB:
    def __init__(self, num_actions):
        self.num_actions = num_actions
        self.action_counts = [0] * num_actions
        self.cumulative_rewards = [0] * num_actions

    def choose_action(self):
        exploration_prob = 0.1  # can play with this
        if random.random() < exploration_prob:
            # explore; random action
            action = random.randint(0, self.num_actions - 1)
        else:
            # exploit; action with the highest estimated reward
            action = self.get_best_action()
        return action

    def update_rewards(self, action, reward):
        # update the cumulative rewards and action counts
        self.cumulative_rewards[action] += reward
        self.action_counts[action] += 1

    def get_best_action(self):
        # choose the action with the highest average reward
        # (in this case lowest prison sentence)
        average_rewards = [reward / max(1, count) for reward, count in zip(self.cumulative_rewards, self.action_counts)]
        return min(range(self.num_actions), key=lambda x: average_rewards[x])

    def get_reward(self, self_action, other_action):
        # normalize the language model's response
        if 'silence' in other_action.lower().strip():
            other_action = 'Silence'
        else:
            other_action = 'Confess'
        if self_action == 0:
            self_action = 'Silence'
        else:
            self_action = 'Confess'
        # rewards
        if ((self_action == 'Silence') and (other_action == 'Silence')):
            return 1
        elif ((self_action == 'Confess') and (other_action == 'Confess')):
            return 3
        elif self_action == 'Silence':
            return 5
        else:
            return 0
        

# todo --> contextual bandit


