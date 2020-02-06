import numpy as np
from policies import base_policy as bp
from itertools import combinations_with_replacement

EPSILON = 0.05
LR = 0.0001
DISCOUNT = 0.5
NUM_VALUES = 11
#STATE_DIM = 1 + NUM_VALUES + (5 * NUM_VALUES)
N_STEPS = 3
STATE_DIM = 4 * N_STEPS * NUM_VALUES

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
        self.weights = np.zeros(STATE_DIM)
#        self.weights = np.random.randn(STATE_DIM)
        self.last_states = []
        self.last_actions = []
        self.last_rewards = []
        self.positions = self.create_positions(N_STEPS)

    def learn(self, round, prev_state, prev_action,
              reward, new_state, too_slow):
        try:
            if round % 100 == 0:
                if round > self.game_duration - self.score_scope:
                    self.log("Rewards in last 100 rounds which counts towards the score: " + str(self.r_sum), 'VALUE')
                else:
                    self.log("Rewards in last 100 rounds: " + str(self.r_sum), 'VALUE')
                self.r_sum = 0
            else:
                self.r_sum += reward

        except Exception as e:
            self.log("Something Went Wrong...", 'EXCEPTION')
            self.log(e, 'EXCEPTION')

        for s, a, r in zip(self.last_states, self.last_actions, self.last_rewards):
            self.update_values(s, a, r)

        self.last_states = []
        self.last_actions = []
        self.last_rewards = []

    def update_values(self, state, action, reward):
        q_values = np.zeros(len(bp.Policy.ACTIONS))
        for a_ind, a in enumerate(bp.Policy.ACTIONS):
            q_values[a_ind] = self.get_qvalue(state, a)
        q_opt = reward + self.discount*q_values.max()
        delta = self.get_qvalue(state, action)
        self.weights += self.lr * (q_opt - delta) * self.get_step_features(state)

    def act(self, round, prev_state, prev_action, reward, new_state, too_slow):
        if round > 0:
            self.last_states.append(prev_state)
            self.last_actions.append(prev_action)
            self.last_rewards.append(reward)

#        # turn off exploration when final score is calculated
#        if round > self.game_duration - self.score_scope:
#            self.epsilon = 0

        if np.random.rand() < self.epsilon:
            return np.random.choice(bp.Policy.ACTIONS)

        return self.get_policy(new_state)

    def get_policy(self, state):

        q_values = np.zeros(3)

        for a_ind, a in enumerate(bp.Policy.ACTIONS):
            q_values[a_ind] = self.get_qvalue(state, a)

        return bp.Policy.ACTIONS[np.argmax(q_values)]

    def get_qvalue(self, state, action):
        board, head = state
        head_pos, direction = head
        next_position = head_pos.move(bp.Policy.TURNS[direction][action])
        new_state = board, (next_position, bp.Policy.TURNS[direction][action])
        return np.dot(self.weights, self.get_step_features(new_state))

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
#                temp_pos = temp_pos.move(bp.Policy.TURNS[direction][step])
                r = temp_pos[0]
                c = temp_pos[1]
                temp_feats[route_ind, board[r, c] + 1] += 1

        feats = np.ones(STATE_DIM)
        feats[:-1] = temp_feats.flatten()

        return feats

    def get_step_features(self, state):

        temp_feats = np.zeros([4 * N_STEPS, NUM_VALUES])

        board, head = state
        head_pos, direction = head

        max_x = board.shape[0]
        max_y = board.shape[1]

        for pos_ind, pos in enumerate(self.positions[direction].keys()):
            for version in self.positions[direction][pos]:
                r = int((head_pos[0] + version[0]) % max_x)
                c = int((head_pos[1] + version[1]) % max_y)
                temp_feats[pos_ind, board[r, c] + 1] += 1

        feats = temp_feats.flatten()

        return feats

    def create_positions(self, n_steps=4):

        moves = {'N_F': ('N', (0, 1)),
                 'N_L': ('W', (-1, 0)),
                 'N_R': ('E', (1, 0)),
                 'E_F': ('E', (1, 0)),
                 'E_L': ('N', (0, 1)),
                 'E_R': ('S', (0, -1)),
                 'S_F': ('S', (0, -1)),
                 'S_L': ('E', (1, 0)),
                 'S_R': ('W', (-1, 0)),
                 'W_F': ('W', (-1, 0)),
                 'W_L': ('S', (0, -1)),
                 'W_R': ('N', (0, 1))}

        positions = {}
        for pos in bp.Policy.TURNS.keys():
            positions[pos] = {}
            for i in range(1, n_steps + 1):
                end_states = []
                combinations = combinations_with_replacement(bp.Policy.ACTIONS, i)
                for comb in combinations:
                    dir = pos
                    x_moves = np.zeros(i)
                    y_moves = np.zeros(i)
                    for ind, act in enumerate(comb):
                        move = moves[f'{dir}_{act}']
                        dir = move[0]
                        x_moves[ind] = move[1][0]
                        y_moves[ind] = move[1][1]
                    end_states.append((np.sum(x_moves), np.sum(y_moves)))

                end_states = list(set(end_states))
                positions[pos][f'{i}_l'] = [p for p in end_states if p[0] < 0 and abs(p[0]) >= abs(p[1])]
                positions[pos][f'{i}_r'] = [p for p in end_states if p[0] > 0 and abs(p[0]) >= abs(p[1])]
                positions[pos][f'{i}_d'] = [p for p in end_states if p[1] < 0 and abs(p[0]) <= abs(p[1])]
                positions[pos][f'{i}_u'] = [p for p in end_states if p[1] > 0 and abs(p[0]) <= abs(p[1])]

        return positions


