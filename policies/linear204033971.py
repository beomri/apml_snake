from policies import base_policy as bp
import numpy as np

EPSILON = 1
LR = 0.01
DISCOUNT = 0.4

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
        self.weights = np.random.random([1, 89])
        self.last_states = []
        self.last_actions = []
        self.last_rewards = []
        self.log(f'epsilon: {self.epsilon}')

    def learn(self, round, prev_state, prev_action, reward, new_state, too_slow):

        try:
            if round % 100 == 0:
                self.log(f'{self.epsilon}')
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
        q_values = np.zeros(3)
        for a_ind, a in enumerate(bp.Policy.ACTIONS):
            q_values[a_ind] = self.get_qvalue(state, a)
        q_opt = reward + self.discount*q_values.max()
        delta = self.get_qvalue(state, action)

        self.weights -= self.lr * (q_opt - delta) * self.get_features(state)

    def act(self, round, prev_state, prev_action, reward, new_state, too_slow):
        if round > 0:
            self.last_states.append(prev_state)
            self.last_actions.append(prev_action)
            self.last_rewards.append(reward)


        board, head = new_state
        head_pos, direction = head
        return np.random.choice(bp.Policy.ACTIONS)
        if np.random.rand() < self.epsilon:
            return np.random.choice(bp.Policy.ACTIONS)

        else:
            a = self.get_policy(new_state)
            print(a)
            return a


    def get_policy(self, state):

        board, head = state
        head_pos, direction = head

        q_values = np.zeros(3)

        for a_ind, a in enumerate(bp.Policy.ACTIONS):
#            next_position = head_pos.move(bp.Policy.TURNS[direction][a])
#            new_state = (board, (next_position, bp.Policy.TURNS[direction][a]))
            q_values[a_ind] = self.get_qvalue(state, a)#self.weights.T @ self.get_features(new_state)

        return bp.Policy.ACTIONS[np.argmax(q_values)]


    def get_qvalue(self, state, action):
        board, head = state
        head_pos, direction = head
        next_position = head_pos.move(bp.Policy.TURNS[direction][action])
        new_state = (board, (next_position, bp.Policy.TURNS[direction][action]))
        return self.weights.T @ self.get_features(new_state)


    def get_features(self, state):
        feats = np.zeros(89)
        feats[-1] = 1

        board, head = state
        head_pos, direction = head

        for ind, a in enumerate(bp.Policy.ACTIONS):
            temp_feats = np.zeros(11)
#            next_position = head_pos.move(bp.Policy.TURNS[direction][a])
            r = head_pos[0]
            c = head_pos[1]
            temp_feats[board[r, c] + 1] = 1
            feats[ind*11:(ind+1)*11] = temp_feats
        last_ind = 32

        forward_region = ['F', 'F', 'F']
        forward_left_region = ['F', 'L', 'F', 'R', 'R', 'L', 'L']
        forward_right_region = ['F', 'R', 'F', 'L', 'L', 'R', 'R']
        right_region = ['R','F','R','R']
        left_region = ['L','F','L','L']

        routes = [forward_region, forward_left_region,
                  forward_right_region, right_region, left_region]

        for route_ind, route in enumerate(routes):
            temp_feats = np.zeros(11)
            temp_pos = head_pos
            temp_pos = temp_pos.move(bp.Policy.TURNS[direction][route[0]])
            for step_ind, step in enumerate(route[1:]):
                temp_pos = temp_pos.move(bp.Policy.TURNS[direction][route[0]])
                r = temp_pos[0]
                c = temp_pos[1]
                temp_feats[board[r, c] + 1] += 1
            feats[(last_ind + route_ind*11):(last_ind + (route_ind+1)*11)] = temp_feats

        return feats[:, np.newaxis]
