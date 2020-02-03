from policies import base_policy as bp
import numpy as np
from keras.models import Model
from keras.layers import Input, Dense
import numbers
from sklearn.preprocessing import OneHotEncoder

EPSILON = 0.05
STATE_DIM = 1 + 11 + (5*11)
NUM_VALUES = 11


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
        self.feature_shape = 67
        self.last_states = []
        self.last_actions = []
        self.last_rewards = []
        self.model = PolicyNetwork(self.feature_shape, bp.Policy.ACTIONS.shape[0], 3)

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

        if round > 0:
            self.last_states.append(prev_state)
            self.last_actions.append(prev_action)
            self.last_rewards.append(reward)

        if np.random.rand() < self.epsilon:
            return np.random.choice(bp.Policy.ACTIONS)

        else:
            probs = self.model.predict(self.get_features(new_state))
            return np.random.choice(bp.Policy.ACTIONS, p=probs)

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
                temp_feats[route_ind + 1, board[r, c] + 1] += 1

        feats = np.ones(STATE_DIM)
        feats[:-1] = temp_feats.flatten()

        return feats


class PolicyNetwork:

    def __init__(self, in_shape, out_shape, n_hidden_layers, n_nodes=64,
                 loss='categorical_crossentropy', optimizer='adam', lr=0.01):

        if isinstance(n_nodes, numbers.Number):
            n_nodes = [n_nodes] * n_hidden_layers

        self.in_shape = in_shape
        self.out_shape = out_shape
        self.n_hidden_layers = n_hidden_layers
        self.n_nodes = n_nodes
        self.loss = loss
        self.optimizer = optimizer
        self.lr = lr

        input = Input(shape=(self.in_shape,))
        in_layer = input
        for i in range(n_hidden_layers):
            out_layer = Dense(n_nodes[i], input_shape=(self.in_shape,), activation='relu')(in_layer)
            in_layer = out_layer
        output = Dense(self.out_shape, input_shape=(self.n_nodes[-1],), activation='softmax')(out_layer)
        model = Model(inputs=input, outputs=output)
        model.compile(loss='categorical_crossentropy',
                      optimizer=optimizer,
                      lr=lr,
                      metrics=['accuracy'])

        self.model = model
        print(self.model.summary())

if __name__ == "__main__":
    n = 200
    x = np.zeros(shape=(n, STATE_DIM))
    y = np.random.randint(0, 3, n)

    for i in range(n):
        x[i] = np.random.randint(0, 6, size=STATE_DIM)

    nn = PolicyNetwork(67, 3, 3)
    nn.model.fit(x=x, y=y, batch_size=32, epochs=10, verbose=1, callbacks=None) # integrate callbacks later!