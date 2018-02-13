import tensorflow as tf
import numpy as np
import h5py

from tf_rllab.core.parameterized import Model

from sklearn.metrics import accuracy_score
from scipy.stats import mode

from tf_rllab.core.flip_gradients import flip_gradient

from tensorflow.python.ops import rnn_cell
from tensorflow.contrib.layers.python.layers import initializers

def pad_traj(X, mpl):
    """
    X is [traj_length x feature_dim]
    """
    t, d = X.shape
    X = np.concatenate([X, np.ones((mpl-t,d)) * np.nan], axis=0)
    return X

def last_relevant(output, length):
    """
    from : https://danijar.com/variable-sequence-lengths-in-tensorflow/
    """
    batch_size = tf.shape(output)[0]
    max_length = tf.shape(output)[1]
    out_size = int(output.get_shape()[2])
    index = tf.range(0, batch_size) * max_length + (length - 1)
    flat = tf.reshape(output, [-1, out_size])
    relevant = tf.gather(flat, index)
    return relevant

def recurrence(cell, l_in, eos):
    rnn_outputs, states = \
        tf.nn.dynamic_rnn(cell,l_in,sequence_length=eos,dtype=tf.float32)
    rnn_output = last_relevant(rnn_outputs, eos)
    return rnn_output

def feedforward(l_in, hidden_sizes, nonlinearity,
                hidden_W_init=tf.contrib.layers.xavier_initializer(), t_w_trainable=True,
                hidden_b_init=tf.zeros_initializer, t_b_trainable=True,
                linear_output = True,
                drop_prob = 0.0,
                start_idx = 0):
    fan_in = l_in.get_shape()[-1]
    for j, fan_out in enumerate(hidden_sizes):
        w = tf.get_variable("w_fc{}".format(start_idx + j), shape=(fan_in,fan_out), dtype=tf.float32,
                            initializer=hidden_W_init, trainable=t_w_trainable)
        b = tf.get_variable("b_fc{}".format(start_idx + j), shape=(fan_out,), dtype=tf.float32,
                            initializer=hidden_b_init, trainable=t_b_trainable)
        l_in = tf.nn.xw_plus_b(l_in, w, b)
        if j != (len(hidden_sizes) - 1):
            l_in = tf.nn.dropout(l_in, 1.-drop_prob)
            l_in = nonlinearity(l_in)
        elif not linear_output:
            l_in = nonlinearity(l_in)
        fan_in = fan_out
    return l_in

#class Buffer(object):
#    """
#    replay buffer used to store and sample obs / actions during learning.
#    """
#    def __init__(self, dim, slices, capacity = 10000):
#        self.x = np.zeros((1,dim))
#        # priority is an arbitrary real number > 0
#        self.p = np.zeros(1).astype('float32')
#        self.slices = slices
#        self.capacity = capacity
#
#    def add(self,X,p):
#        self.x = np.row_stack((self.x,X))
#        self.p = np.concatenate([self.p, p * np.ones(X.shape[0])])
#        if self.x.shape[0] > self.capacity:
#            ix = np.random.choice(np.arange(len(self.p)),size=(self.capacity,),replace=False)
#            self.x = self.x[ix]
#            self.p = self.p[ix]
#        assert self.x.shape[0] == len(self.p)
#
#    def sample(self,n):
#        n = np.minimum(n, self.x.shape[0])
#        p = self.p / np.float(self.p.sum()) # normalize priorities as probabilities
#        ixs = np.random.choice(np.arange(len(p)),size=(n,),replace=False,p=p)
#        samples = self.x[ixs]
#	assert samples.shape[-1] == self.x.shape[-1]
#        return np.split(samples,self.slices,axis=1)

class Encoder(object):
    def __init__(self, encoder_size=128, num_encoder_layers=2, batch_size=1):
        state_dim = 51
        action_dim = 2
        self.z_dim = z_dim = 2

        self.states_encode = tf.placeholder(tf.float32, [batch_size, state_dim], name="states_encode")
        self.actions_encode = tf.placeholder(tf.float32, [batch_size, action_dim], name="actions_encode")

        with tf.variable_scope("encoder") as scope:
            # Create LSTM portion of network
            lstm = rnn_cell.LSTMCell(encoder_size, state_is_tuple=True, initializer=initializers.xavier_initializer())
            self.full_lstm = rnn_cell.MultiRNNCell([lstm] * num_encoder_layers, state_is_tuple=True)
            self.lstm_state = self.full_lstm.zero_state(batch_size, tf.float32)

            # Forward pass
            encoder_input = tf.concat(1, [self.states_encode, self.actions_encode])
            output, self.final_state = seq2seq.rnn_decoder([encoder_input], self.lstm_state, self.full_lstm)
            output = tf.reshape(tf.concat(1, output), [-1, encoder_size])

            # Fully connected layer to latent variable distribution parameters
            W = tf.get_variable("latent_w", [encoder_size, 2*z_dim], initializer=initializers.xavier_initializer())
            b = tf.get_variable("latent_b", [2*z_dim])

            logits = tf.nn.xw_plus_b(output, W, b)

        # Separate into mean and logstd
        self.z_mean, self.z_logstd = tf.split(1, 2, logits)

    def get_params(self):
        L=[v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if
                v.name.split('/')[0] == "encoder"]
        return L

    def load_params(self, exp_name, *args):
        with h5py.File("{}/encoder.h5".format(exp_name),"r") as hf:
            assignments = []
            for param in self.get_params():
                path = param.name
                if path in hf:
                    assignments.append(
                        param.assign(hf[path][...])
                        )
                else:
                    import pdb; pdb.set_trace()
                    halt = True

        sess = tf.get_default_session()
        sess.run(assignments)

    def predict(self, XZ, A, Y=None, **kwargs):
        sess = tf.get_default_session()
        assert XZ.ndim == 3
        X = XZ[...,:-self.z_dim]
        B, T, Do = X.shape
        assert B == 1
        # Initialize the internal state
        state = []
        for c, m in self.lstm_state:
            state.append((c.eval(session= sess), m.eval(session= sess)))

        # Loop over all timesteps, find posterior over z
        for t in range(T):
            # Get state and action values for specific time step
            #s_enc, a_enc = s[:,t], a[:,t]
            s_enc, a_enc = X[:,t,:], A[:,t,:]

            # Construct inputs to network
            feed_in = {}
            feed_in[self.states_encode] = s_enc
            feed_in[self.actions_encode] = a_enc
            for i, (c, m) in enumerate(self.lstm_state):
                feed_in[c], feed_in[m] = state[i]

            # Define outputs
            feed_out = [self.z_mean, self.z_logstd]
            for c, m in self.final_state:
                feed_out.append(c)
                feed_out.append(m)

            # Make pass
            res = sess.run(feed_out, feed_in)
            z_mean = res[0]
            z_logstd = res[1]
            state_flat = res[2:]
            state = [state_flat[i:i+2] for i in range(0, len(state_flat), 2)]

        #return z_mean, z_logstd, state
        #print("predicting")
        return z_mean, (0.0, 0.0)

class BaseRewardModel(object):
    def __init__(self, name, trainer, learning_rate, obs_dim, act_dim, hidden_sizes, hidden_nonlinearity,
                 batch_size=None, drop_prob=0.0, wgan=1, clip=10.0, epochs=20,
                 treat_z = "ignore", **kwargs):

        self.name = name

        self.trainer = trainer
        self.batch_size = batch_size
        self.wgan = wgan
        self.name = name
        self.epochs = epochs
        self.lr = tf.Variable(learning_rate, dtype=tf.float32)
        self.trainer = trainer(self.lr)

        self._treat_z = treat_z
        self._hidden_sizes = hidden_sizes
        self._hidden_nonlinearity = hidden_nonlinearity
        self._drop_prob = drop_prob

        self.treat_z = treat_z
        self._z_dim = kwargs["z_dim"]
        self._z_idx = kwargs["z_idx"]
        self._z_hidden_sizes = kwargs["z_hidden_sizes"]

    def _construct_graph(self, x, a, output_dim=1):

        assert self.treat_z in ["mul","concat","ignore"]

        with tf.variable_scope(self.name):
            # merge x and z by elemwise mul
            if self.treat_z == "mul":
                raise NotImplementedError

            # slice off, and ignore z (i.e., as in standard InfoGAIL)
            if self.treat_z == "ignore":
                l_in = tf.concat(1,[x[:,:-self._z_dim], a])
                model = model = feedforward(l_in, self._hidden_sizes +
                        [output_dim],
                        self._hidden_nonlinearity,
                        tf.contrib.layers.xavier_initializer(), True,
                        tf.zeros_initializer, True,
                        linear_output = True,
                        drop_prob = self._drop_prob)

            # merge x and z by concatenation
            elif self.treat_z == "concat":
                raise NotImplementedError

        return model

    def _construct_output(self):

        if self.wgan:
            self.cost = tf.reduce_mean(self.model * self.y)
            #self.reward = tf.log(tf.exp(self.model) + 1.)
            self.clip_ops = [v.assign(tf.clip_by_value(v, -clip, clip)) for v in self.get_params()]

        else:
            # warning : sigmoid_cross_entropy 's arguments are FLIPPED in
            # documentation!
            self.cost = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(self.model,self.y))
            #self.reward = -tf.log(1.0 - tf.nn.sigmoid(self.model))
            self.clip_ops = []

        self.opt = self.trainer.minimize(self.cost)

    def compute_reward(self, X, X_a):
        sess = tf.get_default_session()
        r = sess.run(self.reward, {self.x:X, self.a:X_a})
        return r

    def compute_score(self, X, X_a):
        sess = tf.get_default_session()
        return sess.run(self.model, {self.x:X,self.a:X_a})

    def train(self, obs_ex, act_ex, obs_pi, act_pi):
        sess = tf.get_default_session()
        n_ = obs_ex.shape[0]
        n = obs_pi.shape[0]
        assert n == n_

        X = np.row_stack([obs_ex, obs_pi])
        A = np.row_stack([act_ex, act_pi])
        if self.wgan:
            # expert is labeled as -1
            Y = np.concatenate([-1. * np.ones(n), np.ones(n)])[...,None]
        else:
            Y = np.concatenate([np.ones(n),np.zeros(n)])[...,None]
        losses = []
        N = 2 * n
        for i in range(self.epochs):
            ixs = np.random.permutation(range(N))
            minibatches = np.split(ixs,N/self.batch_size)
            for mb in minibatches:
                loss, _ = sess.run([self.cost, self.opt], {self.x:X[mb],
                                                           self.y:Y[mb],
                                                           self.a:A[mb]})
                _ = sess.run(self.clip_ops)
                losses.append(loss)

        loss = np.mean(losses)
        return loss

    def get_params(self):
        L=[v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
           if v.name.split('/')[0] == "reward" and v.name.split('/')[1] == self.name]
        return L

class RewardModel(BaseRewardModel):
    def __init__(self, **kwargs):
        obs_dim, act_dim = kwargs['obs_dim'], kwargs['act_dim']

        self.x = tf.placeholder(tf.float32, shape=(None, obs_dim), name="x")
        self.a = tf.placeholder(tf.float32, shape=(None, act_dim), name="a")
        self.y = tf.placeholder(tf.float32, shape=(None,1), name="y")

        super(RewardModel,self).__init__(**kwargs)
        self.model = super(RewardModel,self)._construct_graph(self.x,self.a)

        super(RewardModel,self)._construct_output()

        if self.wgan:
            self.reward = tf.log(tf.exp(self.model) + 1.)
        else:
            self.reward = -tf.log(1.0 - tf.nn.sigmoid(self.model))

        # treat reward as 1D vector
        self.reward = tf.reshape(self.reward,[-1])

class FactoredRewardModel(BaseRewardModel):
    def __init__(self, n_agents, **kwargs):
        obs_dim, act_dim = kwargs['obs_dim'], kwargs['act_dim']

        self.x = tf.placeholder(tf.float32, shape=(None, obs_dim), name="x")
        self.a = tf.placeholder(tf.float32, shape=(None, act_dim), name="a")
        self.y = tf.placeholder(tf.float32, shape=(None, 1), name="y")

        x = tf.reshape(self.x,shape=(-1, obs_dim / n_agents))
        a = tf.reshape(self.a,shape=(-1, act_dim / n_agents))

        super(FactoredRewardModel,self).__init__(**kwargs)

        output_dim = 4
        model = \
            super(FactoredRewardModel,self)._construct_graph(x,a,output_dim=output_dim)

        self.h = h = tf.reshape(model,(-1,n_agents * output_dim))
        self.model = feedforward(h, [32, 1], tf.nn.tanh,
                tf.contrib.layers.xavier_initializer(), True,
                tf.zeros_initializer, True,
                linear_output = True,
                drop_prob = 0.0)
        super(FactoredRewardModel,self)._construct_output()

        if self.wgan:
            self.reward = tf.log(tf.exp(self.model) + 1.)
        else:
            raise NotImplementedError

    def compute_reward(self, X, X_a):
        sess = tf.get_default_session()
        r, h = sess.run([self.reward,self.h], {self.x:X, self.a:X_a})

        return r

class InfoModel(Model):
    """
    model for predicting latent code z, conditioned on (s, a) pair.
    a.k.a, the variational posterior
    """
    save_name = "epochs"

    def get_params(self):
        L=[v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if v.name.split('/')[0] == "info"]
        return L

    def _XZ_to_X_Z(self, XZ):
        X, Z = XZ[...,:-self.z_dim], XZ[...,-self.z_dim:]
        Z = np.argmax(Z,axis=-1)
        return X, Z

    def _train(self, itr, feed_dict, N):
        #feed_dict = self._convert_domain_indices(feed_dict)

        sess = tf.get_default_session()
        losses = []

        # decay learning rate
        sess.run(tf.assign(
            self.lr,
            self.learning_rate * (self.decay_rate ** (itr / self.decay_step))
            ))

        # train over minibatches
        for i in range(self.epochs):
            ixs = np.random.permutation(range(N))
            minibatches = np.array_split(ixs,range(0,N,self.batch_size)[1:])
            for mb in minibatches:
                _feed_dict = {tensor:array[mb] for tensor, array in
                        feed_dict.items()}
                loss, _ = sess.run([self.cost, self.opt], _feed_dict)
                losses.append(loss)

        loss = np.mean(losses)
        return loss

    def _convert_domain_indices(self, feed_dict):
        if self.y in feed_dict.keys():
            try:
                feed_dict[self.y] = np.array([self.domain_indices.index(y) for y in
                    feed_dict[self.y]])
            except:
                import pdb; pdb.set_trace()
        return feed_dict

    def _compute_reward(self, feed_dict):
        #feed_dict = self._convert_domain_indices(feed_dict)
        sess = tf.get_default_session()
        r = sess.run(self.reward, feed_dict)
        return r

    def _predict(self, feed_dict):
        #feed_dict = self._convert_domain_indices(feed_dict)
        sess = tf.get_default_session()

        y_hat = None
        z_hat = sess.run(self.z_hat, feed_dict)
        if self.y in feed_dict.keys():
            y_hat = sess.run(self.y_hat, feed_dict)

        # needs an entropy term ?
        return z_hat, y_hat

    def _create_obj(self, z_discrete):
        if z_discrete:
            # compute log probability of label Q(z | s, a)
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=self.z, logits=self.model)

            # compute entropy term H(z) with monte carlo estimaton
            mc_est = tf.reduce_mean(tf.nn.softmax(self.model), axis = 0)
            entropy = tf.reduce_sum(mc_est * tf.log(mc_est / self.prior))

            # prediction from model
            z_hat = tf.argmax(self.model, axis=self.model.get_shape().ndims-1)
            y_hat = tf.argmax(self.cnf_model, axis=self.model.get_shape().ndims-1)

        else:
            raise NotImplementedError
            cost = tf.l2_loss(self.z - self.model)
            z_hat = self.model

        cost = (self.reg * ((1.-self.cnf) * tf.reduce_mean(cross_entropy))
                + self.ent * entropy)

        if not self.include_cnf_in_reward:
            reward = tf.reshape(-cost, [-1])

        # include this in the reward term?
        if self.cnf > 0.0:
            confusion_cross_entropy = \
                    tf.nn.sparse_softmax_cross_entropy_with_logits(
                            labels=self.y, logits=self.cnf_model
                            )
            cost += self.reg * (self.cnf * tf.reduce_mean(confusion_cross_entropy))

        if self.include_cnf_in_reward:
            reward = tf.reshape(-cost, [-1])

        opt = self.trainer.minimize(cost)

        return cost, reward, entropy, opt, z_hat, y_hat

    @property
    def prior(self):
        x = np.ones(self.z_dim).astype('float32')
        x /= x.sum()
        return tf.constant(x)

class FF_InfoModel(InfoModel):
    """ Feed forward model
    """
    def __init__(self, name, trainer, learning_rate, obs_shape, act_dim, z_dim,
                 y_dim, hidden_sizes, hidden_nonlinearity,
                 batch_size=None, drop_prob=0.0, epochs=20,
                 z_discrete=True, reg=1., ent=0., cnf=0.,
                 decay_rate = 1.0, decay_step = 1,
                 stochastic = False,
                 cnf_hidden_sizes = [],
                 include_cnf_in_reward = False,
                 **kwargs):

        self.x = tf.placeholder(tf.float32, shape=(None, np.prod(obs_shape)-z_dim),name="x")
        self.a = tf.placeholder(tf.float32, shape=(None, act_dim),name="a")
        self.y = tf.placeholder(tf.int32, shape=(None,), name="y")
        if z_discrete:
            self.z = tf.placeholder(tf.int32, shape=(None,),name="z")
        else:
            self.z = tf.placeholder(tf.float32, shape=(None, z_dim),name="z")

        self.batch_size = batch_size
        self.name = name
        self.epochs = epochs
        self.domain_indices = kwargs['domain_indices']

        # objective weights
        self.reg = reg
        self.ent = ent
        self.cnf = cnf
        self.z_dim = z_dim
        self.stochastic = stochastic
        self.include_cnf_in_reward = include_cnf_in_reward

        # learning hyperparam
        self.decay_rate = decay_rate
        self.decay_step = decay_step
        self.learning_rate = learning_rate
        self.lr = tf.Variable(learning_rate, dtype=tf.float32)
        self.trainer = trainer(self.lr)

        with tf.variable_scope(name):
            l_in = tf.concat(1,[self.x, self.a])
            features = feedforward(l_in, hidden_sizes, hidden_nonlinearity,
                    tf.contrib.layers.xavier_initializer(), True,
                    tf.zeros_initializer, True,
                    linear_output = False,
                    drop_prob = drop_prob)

            with tf.variable_scope("output"):
                self.model = feedforward(features, [z_dim], hidden_nonlinearity,
                        tf.contrib.layers.xavier_initializer(), True,
                        tf.zeros_initializer, True,
                        linear_output = True,
                        drop_prob = drop_prob)

            with tf.variable_scope("cnf"):
                flip_feats = flip_gradient(features)
                self.cnf_model = feedforward(
                        flip_feats, cnf_hidden_sizes + [y_dim], hidden_nonlinearity,
                        tf.contrib.layers.xavier_initializer(), True,
                        tf.zeros_initializer, True,
                        linear_output = True,
                        drop_prob = drop_prob)

            self.output = tf.nn.softmax(self.model)

        self.cost, self.reward, self.entropy, self.opt, self.z_hat, self.y_hat = self._create_obj(z_discrete)

    def train(self, itr, XZ, A, Y, **kwargs):
        sess = tf.get_default_session()
        X, Z = self._XZ_to_X_Z(XZ)
        N = X.shape[0]
        N_ = Z.shape[0]
        assert N == N_

        assert not np.isnan(X.sum() + Z.sum())
        feed_dict = {self.x: X, self.z: Z, self.a: A}
        if Y is not None:
            feed_dict[self.y] = Y
        loss = self._train(itr, feed_dict, N)
        return loss

    def compute_reward(self, XZ, A, Y, **kwargs):
        X, Z = self._XZ_to_X_Z(XZ)
        r = self._compute_reward({self.x:X, self.z:Z, self.a:A, self.y:Y})
        return r

    def predict(self, XZ, A, Y=None, **kwargs):
        X, Z = self._XZ_to_X_Z(XZ)
        if X.ndim == 3:
            # suppose there's only one trajectory
            assert XZ.shape[0] == 1
            X = X.reshape(-1,X.shape[-1])
            #Z = Z.reshape(-1,Z.shape[-1])
            A = A.reshape(-1,A.shape[-1])
            Z = np.squeeze(Z)

        assert not np.isnan(X.sum() + Z.sum())
        feed_dict = {self.x: X, self.a: A}
        if Y is not None:
            feed_dict[self.y] = Y
        z_hat, y_hat = self._predict(feed_dict)

        z_acc = accuracy_score(Z, z_hat)
        if Y is not None:
            y_acc = accuracy_score(Y.astype('int32'), y_hat)
        else:
            y_acc = None
        if XZ.ndim == 3:
            z_hat, _ = mode(z_hat, axis= 0)

        return z_hat, (z_acc, y_acc)

    @property
    def is_recurrent(self):
        return False

class RE_InfoModel(InfoModel):
    """
    Recurrent
    """
    def __init__(self, name, trainer, learning_rate, obs_shape, act_dim, z_dim,
                    y_dim, hidden_sizes, hidden_nonlinearity,
                    max_path_length = 20, batch_size=None, drop_prob=0.0,
                    epochs=20, z_discrete=True, reg = 1., ent = 0., cnf = 0.,
                    decay_rate = 1.0, decay_step = 1,
                    stochastic = False,
                    cnf_hidden_sizes=[],
                    include_cnf_in_reward=False,
                    **kwargs):
        # input variables.
        self.x = tf.placeholder(tf.float32, shape=(None, max_path_length, np.prod(obs_shape) - z_dim),name="x")
        self.a = tf.placeholder(tf.float32, shape=(None, max_path_length, act_dim),name="a")
        self.eos = tf.placeholder(tf.int32, shape=(None,), name="eos")
        #self.r = tf.placeholder(t, Yf.float32, shape=(None,1),name="r")
        self.y = tf.placeholder(tf.int32, shape=(None,),name="y")
        if z_discrete:
            self.z = tf.placeholder(tf.int32, shape=(None,),name="z")
        else:
            self.z = tf.placeholder(tf.float32, shape=(None, z_dim),name="z")

        self.batch_size = batch_size
        self.name = name
        self.epochs = epochs
        self.domain_indices = kwargs['domain_indices']

        self.include_cnf_in_reward = include_cnf_in_reward

        # objective weights
        self.reg = reg
        self.ent = ent
        self.cnf = cnf

        self.z_dim = z_dim
        self.stochastic = stochastic
        self.max_path_length = max_path_length

        # learning hyperparam
        self.decay_rate = decay_rate
        self.decay_step = decay_step
        self.learning_rate = learning_rate
        self.lr = tf.Variable(learning_rate, dtype=tf.float32)
        self.trainer = trainer(self.lr)

        recur_dim = kwargs['recur_dim']
        Cell = kwargs['cell']
        feat_dim = np.prod(obs_shape) - z_dim + act_dim
        cell = Cell(recur_dim, max_path_length)

        # model definition
        l_in_all = tf.concat(2,[self.x, self.a])
        with tf.variable_scope(name):
            feat_out = []
            ## MLP Features
            with tf.variable_scope("feat") as scope:
                for i, l_in in \
                enumerate(tf.split(value=l_in_all,split_dim=1,num_split=max_path_length)):
                    if i > 0:
                        scope.reuse_variables()

                    l_in = tf.reshape(l_in,[-1,feat_dim])
                    feat_out.append(
                            feedforward(l_in, hidden_sizes + [z_dim], hidden_nonlinearity,
                            tf.contrib.layers.xavier_initializer(), True,
                            tf.zeros_initializer, True,
                            linear_output = False,
                            drop_prob = drop_prob)
                            )

            ## Recurrent Encoding
            with tf.variable_scope("output") as scope:
                # rnn cell used for variational posterior
                feat_out = tf.pack(feat_out,1)
                rnn_output = recurrence(cell, feat_out, self.eos)

                self.model = feedforward(
                        rnn_output, [z_dim], None,
                        tf.contrib.layers.xavier_initializer(), True,
                        tf.zeros_initializer, True,
                        linear_output = True,
                        drop_prob = drop_prob
                        )

                # rnn cell (same parameters) used for domain confusion
                flip_feats = flip_gradient(feat_out)
                scope.reuse_variables()
                flip_rnn_output = recurrence(cell, flip_feats, self.eos)

            with tf.variable_scope("cnf"):
                self.cnf_model = feedforward(
                        flip_rnn_output, [y_dim], None,
                        tf.contrib.layers.xavier_initializer(), True,
                        tf.zeros_initializer, True,
                        linear_output = True,
                        drop_prob = drop_prob
                        )

                self.output = tf.nn.softmax(self.model)

        self.cost, self.reward, self.entropy, self.opt, self.z_hat, self.y_hat = self._create_obj(z_discrete)

    def train(self, itr, XZ, A, Y, EOS):
        sess = tf.get_default_session()
        X, Z = self._XZ_to_X_Z(XZ)
        Z = Z[:,0] # labels are repeated across traj
        Y = Y[:,0]
        N = X.shape[0]
        N_ = Z.shape[0]
        assert N == N_

        X = np.nan_to_num(X)
        A = np.nan_to_num(A)
        Z = np.nan_to_num(Z)

        assert not ( np.isnan(X.sum()) + \
                np.isnan(A.sum()) +\
                np.isnan(EOS.sum()))

        feed_dict = {self.x: X, self.z: Z, self.a: A}
        if Y is not None:
            feed_dict[self.y] = Y
        feed_dict.update({self.eos:EOS})
        loss = self._train(itr, feed_dict, N)
        return loss

    def _process_XZA(self, XZ, A, Y):
        if XZ.ndim == 2:
            # format inputs to [1 x traj_len x feature]
            XZ = pad_traj(XZ, self.max_path_length)
            A = pad_traj(A, self.max_path_length)
            X, Z = self._XZ_to_X_Z(XZ)
            # 
            X = X[None,...]
            A = A[None,...]
            Z = np.array([Z[0]]) # single "label" for entire traj
            if Y is not None:
                Y = np.array([Y[0]])
        elif XZ.ndim == 3:
            X, Z = self._XZ_to_X_Z(XZ)
            Z = Z[:,0] # batch of labels
            if Y is not None:
                Y = Y[:,0]
        else:
            raise NotImplementedError, "_process_XZA input with ndim: {}".format(XZ.ndim)

        X = np.nan_to_num(X)
        Z = np.nan_to_num(Z)
        A = np.nan_to_num(A)
        if Y is not None:
            Y = np.nan_to_num(Y)
        return X, Z, A, Y

    def compute_reward(self, XZ, A, Y, EOS):
        """
        Note: Assigns scalar reward to ENTIRE trajectory
        """
        assert XZ.ndim == 2
        X, Z, A, Y = self._process_XZA(XZ, A, Y)

        feed_dict = {self.x:X, self.z:Z, self.a:A, self.y:Y}
        feed_dict.update({self.eos:EOS})
        r = self._compute_reward(feed_dict)
        # TODO: needs an entropy term ?
        return r

    def predict(self, XZ, A, EOS, Y=None):
        assert XZ.ndim == 3
        if XZ.shape[1] < self.max_path_length:
            PAD = np.zeros((XZ.shape[0],
                    self.max_path_length - XZ.shape[1],
                    XZ.shape[2]))
            XZ = np.concatenate([XZ, PAD], axis = 1)
            PAD = np.zeros((A.shape[0],
                    self.max_path_length - A.shape[1],
                    A.shape[2]))
            A = np.concatenate([A, PAD], axis = 1)
        X, Z, A, Y = self._process_XZA(XZ, A, Y)
        assert not ( np.isnan(X.sum()) + \
                np.isnan(A.sum()) +\
                np.isnan(EOS.sum())
                )

        feed_dict = {self.x:X, self.a:A}
        if Y is not None:
            feed_dict[self.y] = Y
        feed_dict.update({self.eos:EOS})

        z_hat, y_hat = self._predict(feed_dict)
        z_acc = accuracy_score(Z, z_hat)
        if Y is not None:
            Y = Y.astype('int32')
            y_acc = accuracy_score(Y, y_hat)
        else:
            y_acc = None

        return z_hat, (z_acc, y_acc)

    @property
    def is_recurrent(self):
        return True


class InverseDynamicsModel(object):
    def __init__(self, name, trainer, obs_shape, act_dim, hidden_sizes, hidden_nonlinearity,
            active_epoch = 0, batch_size = None, epochs = 20, T = 1.0):
        self.x = tf.placeholder(tf.float32, shape=(batch_size,obs_shape),name="x")
        self.x_prime = tf.placeholder(tf.float32, shape=(batch_size,obs_shape),name="x_prime")
        self.a = tf.placeholder(tf.float32, shape=(batch_size,act_dim),name="a")
        self.trainer = trainer
        self.batch_size = batch_size
        self.epochs = epochs
        self.active_epoch = active_epoch
        self.T = T
        self.all_losses = []
        self.buff = Buffer(obs_shape + obs_shape + act_dim,
                           np.cumsum([obs_shape,obs_shape,act_dim])[:-1]
                           )
        l_in = tf.concat(1,[self.x, self.x_prime])
        with tf.variable_scope(name):
            self.model = model = feedforward(l_in, hidden_sizes + [act_dim], hidden_nonlinearity,
                    tf.contrib.layers.xavier_initializer(), True,
                    tf.zeros_initializer, True,
                    linear_output = True,
                    drop_prob = 0.0)
            self.cost = tf.reduce_mean(tf.nn.l2_loss(model - self.a))
            self.opt = self.trainer.minimize(self.cost)

    def predict(self,X):
        sess = tf.get_default_session()
        B, T, F = X.shape

        X_before = X[:,:-1,:]
        X_after = X[:,1:,:]
        X_before = np.reshape(X_before,(B * (T-1), F))
        X_after = np.reshape(X_after,(B * (T-1), F))
        assert X_before.shape[0] == X_after.shape[0]
        A = sess.run(self.model,{self.x:X_before, self.x_prime:X_after})
        _, Fa = A.shape
        A = np.reshape(A,(B,T-1,Fa))
        return A

    def relabel(self,X,A,loss):
        """
        """
        pred_A = self.predict(X)
        # do not discard final timestep
        pred_A = np.concatenate([pred_A, np.expand_dims(A[:,-1,:],axis=1)],axis=1)
        #A = A[:,:-1,:]
        assert np.allclose(A.shape,pred_A.shape)
        p = np.random.binomial(1, 1.0/np.exp(loss / self.T), pred_A.shape[0])
        AA = np.concatenate((A[None,...],pred_A[None,...]),axis=0)
        A_ = AA[p,np.arange(len(p))]
        return A_, np.mean(p)

    def train(self, X, X_prime, A):
        assert np.ndim(X) == 2
        sess = tf.get_default_session()
        n_ = X.shape[0]
        n = X_prime.shape[0]
        n__ = A.shape[0]
        batch_size = self.batch_size
        if batch_size is None:
            batch_size = 10
        assert (n == n_) and (n == n__)
        losses = []
        for i in range(self.epochs):
            ixs = np.random.permutation(range(n))
            minibatches = np.split(ixs,n/batch_size)
            for mb in minibatches:
                X_batch = X[mb]
                X_prime_batch = X_prime[mb]
                A_batch = A[mb]

                _, loss_disc = sess.run([self.opt,self.cost],{self.x:X_batch,self.x_prime:X_prime_batch,self.a:A_batch})
                losses.append(loss_disc)
        l_d = np.mean(losses)
        self.all_losses.append(l_d)
        return l_d

    @property
    def average_loss(self):
        return np.mean(self.all_losses)

if __name__ == "__main__":
    trainer = tf.train.AdamOptimizer()
    cell = tf.nn.rnn_cell.GRUCell
    obs_shape = (51,)
    act_dim = 3
    z_dim = 2
    hidden_sizes = [12,12]
    hidden_nonlinearity = tf.nn.tanh
    model= RE_InfoModel("info", trainer, obs_shape, act_dim, z_dim, hidden_sizes, hidden_nonlinearity,
                 max_path_length = 50, batch_size=None, drop_prob=0.0, wgan=1, clip=10.0, epochs=20,
                 z_discrete=True, info_reg = 1., cell= cell, recur_dim= 128)
    halt = True

