from policies import base_policy as bp
import numpy as np

EPSILON = 0.05
LR = 0.001
DISCOUNT = 0.15

class Linear204033971(bp.Policy):
    """
    A policy which avoids collisions with obstacles and other snakes. It has an epsilon parameter which controls the
    percentag of actions which are randomly chosen.
    """

    def cast_string_args(self, policy_args):
        policy_args['epsilon'] = float(policy_args['epsilon']) if 'epsilon' in policy_args else EPSILON
        policy_args['lr'] = float(policy_args['lr']) if 'lr' in policy_args else LR
        policy_args['discount'] = float(policy_args['discount']) if 'discount' in policy_args else DISCOUNT
        return policy_args

    def init_run(self):
        self.r_sum = 0
        self.weights = np.zeros([89, 1])
        self.last_states = []
        self.last_actions = []
        self.last_rewards = []
#        self.log(f'epsilon: {self.epsilon}')

    def learn(self, round, prev_state, prev_action, reward, new_state, too_slow):

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
        self.weights += self.lr * (q_opt - delta) * self.get_features(state)


    def act(self, round, prev_state, prev_action, reward, new_state, too_slow):
        if round > 0:
            self.last_states.append(prev_state)
            self.last_actions.append(prev_action)
            self.last_rewards.append(reward)

        board, head = new_state
        head_pos, direction = head
#        return np.random.choice(bp.Policy.ACTIONS)
    
        if np.random.rand() < self.epsilon:
            return np.random.choice(bp.Policy.ACTIONS)
        
        return self.get_policy(new_state)


    def get_policy(self, state):

        board, head = state
        head_pos, direction = head

        q_values = np.zeros(3)

        for a_ind, a in enumerate(bp.Policy.ACTIONS):
            q_values[a_ind] = self.get_qvalue(state, a)

        return bp.Policy.ACTIONS[np.argmax(q_values)]


    def get_qvalue(self, state, action):
        board, head = state
        head_pos, direction = head
        next_position = head_pos.move(bp.Policy.TURNS[direction][action])
        new_state = (board, (next_position, bp.Policy.TURNS[direction][action]))
        return np.squeeze(self.weights.T @ self.get_features(new_state))


    def get_features(self, state):
        feats = np.zeros(89)
        feats[-1] = 1

        board, head = state
        head_pos, direction = head

        temp_feats = np.zeros(11)
        r = head_pos[0]
        c = head_pos[1]
        temp_feats[board[r, c] + 1] = 1
        feats[0:11] = temp_feats
        last_ind = 11

        forward_region = ['F', 'F', 'F']
        forward_left_region = ['F', 'L', 'F', 'R', 'R', 'L', 'L']
        forward_right_region = ['F', 'R', 'F', 'L', 'L', 'R', 'R']
        right_region = ['R', 'F', 'R', 'R']
        left_region = ['L', 'F', 'L', 'L']

        routes = [forward_region, forward_left_region,
                  forward_right_region, right_region, left_region]

        for route_ind, route in enumerate(routes):
            temp_feats = np.zeros(11)
            temp_pos = head_pos
            temp_pos = temp_pos.move(bp.Policy.TURNS[direction][route[0]])
            for step in route[1:]:
                temp_pos = temp_pos.move(bp.Policy.TURNS[direction][step])
                r = temp_pos[0]
                c = temp_pos[1]
                temp_feats[board[r, c] + 1] += 1
            feats[(last_ind + route_ind*11):(last_ind + (route_ind+1)*11)] = temp_feats
#            self.log(f'{np.arange((last_ind + route_ind*11),(last_ind + (route_ind+1)*11))}')

        return feats[:, np.newaxis]
