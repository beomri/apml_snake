from policies import base_policy as bp
import numpy as np

EPSILON = 0.05


class Custom204033971(bp.Policy):
    """
    A policy which avoids collisions with obstacles and other snakes. It has an epsilon parameter which controls the
    percentag of actions which are randomly chosen.
    """

    def cast_string_args(self, policy_args):
        policy_args['epsilon'] = float(policy_args['epsilon']) if 'epsilon' in policy_args else EPSILON
        return policy_args

    def init_run(self):
        self.r_sum = 0
        self.weights = np.zeros([67, 1])
        self.last_states = []
        self.last_actions = []
        self.last_rewards = []
        self.model = PolicyNetwork(bp.Policy.)

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

    def act(self, round, prev_state, prev_action, reward, new_state, too_slow):

        board, head = new_state
        head_pos, direction = head

        if np.random.rand() < self.epsilon:
            return np.random.choice(bp.Policy.ACTIONS)

        else:
            for a in list(np.random.permutation(bp.Policy.ACTIONS)):

                # get a Position object of the position in the relevant direction from the head:
                next_position = head_pos.move(bp.Policy.TURNS[direction][a])
                r = next_position[0]
                c = next_position[1]

                # look at the board in the relevant position:
                if board[r, c] > 5 or board[r, c] < 0:
                    return a

            # if all positions are bad:
            return np.random.choice(bp.Policy.ACTIONS)

            self.model.predict

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
            feats[(last_ind + route_ind * 11):(last_ind + (route_ind + 1) * 11)] = temp_feats
        #            self.log(f'{np.arange((last_ind + route_ind*11),(last_ind + (route_ind+1)*11))}')

        return feats[:, np.newaxis]


class PolicyNetwork:

    def __init__(self, in_shape, out_shape, n_hidden_layers, n_nodes=64,
                 loss='categorical_crossentropy', optimizer='adam', lr=0.01):

        self.in_shape = in_shape
        self.out_shape = out_shape
        self.n_layers = n_hidden_layers

        input = Input(shape=(self.in_shape,))
        in_layer = input

        for i in n_hidden_layers:
            out_layer = Dense(n_nodes, activation='relu')(in_layer)
            in_layer = out_layer
        output = Dense(self.out_shape, activation='softmax')(out_layer)
        model = Model(inputs=input, outputs=output)
        model.compile(loss='categorical_crossentropy',
                      optimizer=optimizer,
                      lr=lr,
                      metrics=['accuracy'])

        self.model = model

