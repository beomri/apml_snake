import numpy as np
from policies import base_policy as bp


EPSILON = 0.05
LR = 0.01
DISCOUNT = 0.5
NUM_VALUES = 11
STATE_DIM = 1 + NUM_VALUES + (5 * NUM_VALUES)


class Linear204033971(bp.Policy):
    """
    A policy which avoids collisions with obstacles and other snakes.
    It has an epsilon parameter which controls the
    percentag of actions which are randomly chosen.
    """

    def cast_string_args(self, policy_args):
        policy_args['epsilon'] = float(policy_args['epsilon']) if 'epsilon' in policy_args else EPSILON
        policy_args['lr'] = float(policy_args['lr']) if 'lr' in policy_args else LR
        policy_args['discount'] = float(policy_args['discount']) if 'discount' in policy_args else DISCOUNT
        return policy_args

    def init_run(self):
        self.r_sum = 0
        self.weights = np.zeros([3, STATE_DIM])
        self.last_states = []
        self.last_actions = []
        self.last_rewards = []
        self.act2ind = {a: i for i, a in enumerate(bp.Policy.ACTIONS)}

    def learn(self, round, prev_state, prev_action,
              reward, new_state, too_slow):

        for s, a, r in zip(self.last_states, self.last_actions, self.last_rewards):
            self.update_values(s, a, r)

        self.last_states = []
        self.last_actions = []
        self.last_rewards = []

    def update_values(self, state, action, reward):
        q_values = self.get_qvalues(state)
        q_opt = reward + (self.discount * q_values.max())
        delta = q_values[self.act2ind[action]]
        self.weights[self.act2ind[action], :] += self.lr * (q_opt - delta) * self.get_features(state)

    def act(self, round, prev_state, prev_action, reward, new_state, too_slow):
        if round > 0:
            self.last_states.append(prev_state)
            self.last_actions.append(prev_action)
            self.last_rewards.append(reward)

        # turn off exploration when final score is calculated
        if round > (self.game_duration - self.score_scope):
            self.epsilon = 0

        if np.random.rand() < self.epsilon:
            return np.random.choice(bp.Policy.ACTIONS)

        return self.get_policy(new_state)

    def get_policy(self, state):
        return bp.Policy.ACTIONS[np.argmax(self.get_qvalues(state))]

    def get_qvalues(self, state):
        return self.weights @ self.get_features(state)

    def get_features(self, state):
        temp_feats = np.zeros([6, NUM_VALUES])

        board, head = state
        head_pos, direction = head

        r = head_pos[0]
        c = head_pos[1]
        temp_feats[-1, board[r, c] + 1] = 1

        forward_region = ['F', 'F', 'F']
        forward_left_region = ['L', 'R', 'F', 'F', 'L', 'L', 'F']
        forward_right_region = ['R', 'L', 'F', 'F', 'R', 'R', 'F']
        right_region = ['R', 'F', 'R', 'R']
        left_region = ['L', 'F', 'L', 'L']

        routes = [forward_region, forward_left_region,
                  forward_right_region, right_region, left_region]

        for route_ind, route in enumerate(routes):
            temp_pos = head_pos
            temp_direction = direction
            for step in route:
                temp_direction = bp.Policy.TURNS[temp_direction][step]
                temp_pos = temp_pos.move(temp_direction)
                r = temp_pos[0]
                c = temp_pos[1]
                temp_feats[route_ind, board[r, c] + 1] += 1

        feats = np.ones(STATE_DIM)
        feats[:-1] = temp_feats.flatten()

        return feats
