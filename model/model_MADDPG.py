import tensorflow as tf
import tensorflow.contrib as tc
import numpy as np
import env

class MADDPG():
    def __init__(self, name, layer_norm=True, nb_actions=7, nb_input=4, nb_other_action=7*99):
        gamma = 0.999
        self.layer_norm = layer_norm
        self.nb_actions = nb_actions
        #BSlist = env.PossibleBS_Index(index, env.BS_Position)
        state_input = tf.placeholder(shape=[None, nb_input], dtype=tf.float32)
        action_input = tf.placeholder(shape=[None, nb_actions], dtype=tf.float32)
        other_action_input = tf.placeholder(shape=[None, nb_other_action], dtype=tf.float32)
        reward = tf.placeholder(shape=[None, 1], dtype=tf.float32)

        def actor_network(name):

            with tf.variable_scope(name) as scope:
                init_W = tf.contrib.layers.xavier_initializer()
                init_b = tf.constant_initializer(0.01)

                s = state_input    # (?, 4)
                l1 = tf.layers.dense(s, 64, activation = tf.nn.relu6,
                                    kernel_initializer=init_W, bias_initializer=init_b, trainable=True)
                action = tf.layers.dense(l1, self.nb_actions,
                                    kernel_initializer=init_W, trainable=True, activation=None)
            return action

        def critic_network(name_, action_input, reuse=False):
            #tf.reset_default_graph()
            with tf.variable_scope(name_) as scope:
                if reuse:
                    scope.reuse_variables()
                init_W = tf.contrib.layers.xavier_initializer()
                init_b = tf.constant_initializer(0.01)

                s = state_input # ?, 4
                l1 = tf.layers.dense(s, 64, activation = tf.nn.relu6, kernel_initializer=init_W, bias_initializer=init_b, trainable=True)
                cl = tf.concat([l1, action_input], axis=-1)
                Q = tf.layers.dense(cl, 1, kernel_initializer=init_W, trainable=True, bias_initializer=init_b)
            return Q

        self.action_output = actor_network(name + '_actor')
        self.critic_output = critic_network(name + '_critic', action_input=tf.concat([action_input, other_action_input], axis=1))
        self.state_input = state_input
        self.action_input = action_input
        self.other_action_input = other_action_input
        self.reward = reward

        self.actor_optimizer = tf.train.AdamOptimizer(1e-4)
        self.critic_optimizer = tf.train.AdamOptimizer(1e-3)

        # base station
        tf.cast(tf.clip_by_value(self.action_output[:,0], 0, env.h_params.NUM_MABS + env.h_params.NUM_MIBS + env.h_params.NUM_PBS),
                tf.int32)
        # number of channels
        tf.cast(tf.clip_by_value(self.action_output[:,1], 0, 6), tf.int32)
        # which channel
        tf.cast(tf.clip_by_value(self.action_output[:,2:7], 0, self.action_output[:,1]), tf.int32)

        """
        ao = self.action_output[:,0]
        min_BS_action = min(ao)
        ao += min_BS_action
        max_BS_action = max(ao)
        clipped_value_BS = int(env.VUE_NUM*ao/max_BS_action)
        tf.assign(self.action_output[:, 0], clipped_value_BS)

        ac = self.action_output[:,1]
        min_chan_action = min(ac)
        ac += min_chan_action
        max_chan_action = max(ac)
        clipped_value_chan = int(env.Channels_PBS_NUM*ac/max_chan_action)
        tf.assign(self.action_output[:, 1], clipped_value_chan)
        """
        self.actor_loss = -tf.reduce_mean(critic_network(name + '_critic', action_input=tf.concat([self.action_output, self.other_action_input], axis=1), reuse=True))
        self.actor_train = self.actor_optimizer.minimize(self.actor_loss)

        # avs = self.actor_optimizer.compute_gradients(self.actor_loss)
        # aapped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in avs if grad is not None]
        # self.actor_train = self.actor_optimizer.apply_gradients(aapped_gvs)

        self.target_Q = tf.placeholder(shape=[None, 1], dtype=tf.float32)
        self.critic_loss = tf.reduce_mean(tf.square(self.target_Q - self.critic_output))
        self.critic_train = self.critic_optimizer.minimize(self.critic_loss)

        # cvs = self.critic_optimizer.compute_gradients(self.critic_loss)
        # capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in cvs if grad is not None]
        # self.critic_train = self.actor_optimizer.apply_gradients(capped_gvs)

    def train_actor(self, state, other_action, sess):
        sess.run(self.actor_train, {self.state_input: state, self.other_action_input: other_action})

    def train_critic(self, state, action, other_action, target, sess):
        sess.run(self.critic_train, {self.state_input: state, self.action_input: action, self.other_action_input: other_action,
                                     self.target_Q: target})

    def action(self, state, sess):
        return sess.run(self.action_output, {self.state_input: state})

    def Q(self, state, action, other_action, sess):
        return sess.run(self.critic_output, {self.state_input: state, self.action_input: action, self.other_action_input: other_action})
