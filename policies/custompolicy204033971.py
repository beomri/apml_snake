import os
import numpy as np
from policies import base_policy as bp
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LeakyReLU, BatchNormalization
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
#from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

#tf.get_logger().setLevel('ERROR')  # don't print warnings (works in tf 2)


NUM_VALUES = 11
STATE_DIM = 1 + NUM_VALUES + (5 * NUM_VALUES)
EPSILON = 0.05
LR = 0.001
DISCOUNT = 0.15
NODES = 64
HIDDEN = 3
OPT = Adam

class Custom204033971(bp.Policy):
    """
    A policy which avoids collisions with obstacles and other snakes.
    It has an epsilon parameter which controls the
    percentag of actions which are randomly chosen.
    """

    def cast_string_args(self, policy_args):
        policy_args['discount'] = float(policy_args['discount']) if 'discount' in policy_args else DISCOUNT
        policy_args['lr'] = float(policy_args['lr']) if 'lr' in policy_args else LR
        policy_args['hidden'] = float(policy_args['hidden']) if 'hidden' in policy_args else HIDDEN
        policy_args['nodes'] = float(policy_args['nodes']) if 'nodes' in policy_args else NODES
        policy_args['opt'] = float(policy_args['opt']) if 'opt' in policy_args else OPT
        return policy_args

    def init_run(self):
        self.r_sum = 0
        self.last_states = []
        self.last_actions = []
        self.last_rewards = []
        self.pn = PolicyNetwork(STATE_DIM,
                                len(bp.Policy.ACTIONS),
                                self.hidden,
                                self.nodes,
                                optimizer=self.opt,
                                lr=self.lr)
        self.act_dict = {a:n for n,a in enumerate(bp.Policy.ACTIONS)}


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

        X = np.array(self.last_states)
        Y = np.array([self.act_dict[a] for a in self.last_actions])

        SW = np.array(self.last_rewards) * (self.discount ** np.arange(len(self.last_rewards)))

        self.pn.train(X, Y, SW)

        self.last_states = []
        self.last_actions = []
        self.last_rewards = []

    def act(self, round, prev_state, prev_action, reward, new_state, too_slow):

        if round > 0:
            self.last_states.append(self.get_features(prev_state))
            self.last_actions.append(prev_action)
            self.last_rewards.append(reward)

        feats = self.get_features(new_state)
        weights = np.squeeze(self.pn.get_actions(feats))
        new_action = np.random.choice(bp.Policy.ACTIONS, p=weights)
        return new_action


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
                 loss='sparse_categorical_crossentropy',
                 optimizer=Adam, lr=0.0001):

        n_nodes = [n_nodes] * n_hidden_layers

        self.in_shape = in_shape
        self.out_shape = out_shape
        self.n_hidden_layers = n_hidden_layers
        self.n_nodes = n_nodes
        self.loss = loss
        self.optimizer = optimizer(lr=lr)
        self.lr = lr

        input = Input(shape=(self.in_shape,))
        in_layer = input
        for i in range(n_hidden_layers):
            out_dense = Dense(n_nodes[i], input_shape=(self.in_shape,))(in_layer)
            out_activation = LeakyReLU()(out_dense)
            in_layer = out_activation
        output = Dense(self.out_shape, input_shape=(self.n_nodes[-1],),
                       activation='softmax')(out_activation)
        model = Model(inputs=input, outputs=output)
        model.compile(loss=self.loss,
                      optimizer=self.optimizer,
                      metrics=['accuracy'])
        n = 1
        x = np.zeros(shape=(n, STATE_DIM))
        y = np.random.randint(0, 3, n)

        for i in range(n):
            x[i] = np.random.randint(0, 6, size=STATE_DIM)
        model.fit(x, y, verbose=False)
        model.predict(x)
        self.model = model

    def train(self, features, actions, rewards):
        self.model.fit(x=features, y=actions, sample_weight=rewards, verbose=False)

    def get_actions(self, features):
        return self.model.predict(features[np.newaxis, :])

