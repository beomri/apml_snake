from policies import base_policy as bp
import numpy as np

EPSILON = 0.05
LR = 0.001
DISCOUNT = 0.15
STATE_DIM = 1 + 11 + (5*11)
NUM_VALUES = 11

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
        self.weights = np.zeros(STATE_DIM)
#        self.weights = self.get_init_weights()
        self.last_states = []
        self.last_actions = []
        self.last_rewards = []
#        self.log(f'epsilon: {self.epsilon}')

    def get_init_weights(self):
        return np.array([0.        , -0.03092003,  0.        ,  0.        ,  0.        ,
        0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
        0.        ,  0.05040398, -0.03560333, -0.00933181, -0.01785775,
        0.        ,  0.        , -0.02053777, -0.00573837,  0.00141165,
       -0.02458667,  0.        , -0.01469611, -0.17184911, -0.0208431 ,
       -0.06167063,  0.        ,  0.        , -0.04904329,  0.06140326,
        0.06963069,  0.0015481 ,  0.        , -0.03089911, -0.14410397,
       -0.01899338, -0.05495567,  0.        ,  0.        , -0.07919307,
        0.08078631,  0.02967739,  0.03216132,  0.        ,  0.02570092,
       -0.07041553,  0.00109382, -0.03418993,  0.        ,  0.        ,
       -0.04697385,  0.00512719,  0.00195824,  0.02493904,  0.        ,
        0.07211   , -0.08957441, -0.0142287 , -0.0228862 ,  0.        ,
        0.        , -0.08987339, -0.00675773,  0.06915907, -0.01070873,
        0.        , -0.03092003])

    def learn(self, round, prev_state, prev_action, reward, new_state, too_slow):

        try:
            if round % 100 == 0:
                if round > self.game_duration - self.score_scope:
                    self.log("Rewards in last 100 rounds which counts towards the score: " + str(self.r_sum), 'VALUE')
#                    np.save(f'linear_weights/{round}.npy', self.weights)
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
        return np.dot(self.weights, self.get_features(new_state))


    def get_features(self, state):
        temp_feats = np.zeros([6, NUM_VALUES])

        board, head = state
        head_pos, direction = head

        r = head_pos[0]
        c = head_pos[1]
        temp_feats[0, board[r, c] + 1] = 1

        forward_region = ['F', 'F', 'F']
        forward_left_region = ['F', 'L', 'F', 'R', 'R', 'L', 'L']
        forward_right_region = ['F', 'R', 'F', 'L', 'L', 'R', 'R']
        right_region = ['R', 'F', 'R', 'R']
        left_region = ['L', 'F', 'L', 'L']

        routes = [forward_region, forward_left_region,
                  forward_right_region, right_region, left_region]

        for route_ind, route in enumerate(routes):
            temp_pos = head_pos
            temp_pos = temp_pos.move(bp.Policy.TURNS[direction][route[0]])
            for step in route[1:]:
                temp_pos = temp_pos.move(bp.Policy.TURNS[direction][step])
                r = temp_pos[0]
                c = temp_pos[1]
                temp_feats[route_ind+1, board[r, c] + 1] += 1

        feats = np.ones(STATE_DIM)
        feats[:-1] = temp_feats.flatten()

        return feats
