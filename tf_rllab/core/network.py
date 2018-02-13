import tf_rllab.core.layers as L
import tensorflow as tf
import numpy as np
import itertools
from rllab.core.serializable import Serializable
from tf_rllab.core.parameterized import Parameterized, Model
from tf_rllab.core.layers_powered import LayersPowered

from rllab.baselines.base import Baseline
from rllab.misc.overrides import overrides

from tf_rllab.optimizers.first_order_optimizer import Solver
from tf_rllab.optimizers.lbfgs_optimizer import LbfgsOptimizer

from flip_gradients import flip_gradient

class NeuralNetwork(Model):

    def _predict(self, t, X):
        sess = tf.get_default_session()

        N, _ = X.shape
        B = self.input_var.get_shape()[0].value

        if B is None or B == N:
            pred = sess.run(t, {self.input_var: X})
        else:
            pred = [sess.run(t, {self.input_var: X[i:i+B]}) for i in range(0,N,B)]
            pred = np.row_stack(pred)

        return pred

    def likelihood_loss(self):
        if self.output_layer.nonlinearity == tf.nn.softmax:
            logits = self.output_layer.get_logits_for(L.get_output(self.layers[-2]))
            loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(logits, tf.squeeze(self.target_var))
                )

        elif self.output_layer.nonlinearity == tf.identity:
            outputs = self.output_layer.get_output_for(L.get_output(self._layers[-2]))
            loss = tf.reduce_mean(
                0.5 * tf.square(outputs - self.target_var), name='like_loss'
            )

        elif self.output_layer.nonlinearity == tf.nn.sigmoid:

            logits = self.output_layer.get_logits_for(L.get_output(self.layers[-2]))
            sigmoid_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits, tf.squeeze(self.target_var))

            if sigmoid_loss.get_shape().ndims == 2:
                loss = tf.reduce_mean(
                    tf.reduce_sum(sigmoid_loss, reduction_indices= 1)
                )
            else:
                loss = tf.reduce_mean(sigmoid_loss)

        return loss

    def complexity_loss(self, reg, cmx):
        """
        Compute penalties for model complexity (e.g., l2 regularization, or kl penalties for vae and bnn).
        """
        # loss coming from weight regularization
        loss = reg * tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

        # loss coming from data-dependent regularization
        for layer in self.layers:
            if layer.penalize_complexity:
                z_mu, z_sig = layer.get_dparams_for(L.get_output(layer.input_layer))
                d_loss = layer.bayesreg.activ_kl(z_mu,z_sig)

                loss += cmx * d_loss

        return reg * loss

    def loss(self, reg= 0.0, cmx= 1.0):
        return tf.add(self.likelihood_loss(), self.complexity_loss(reg, cmx),name= 'loss')

    @property
    def input_layer(self):
        return self._l_in

    @property
    def output_layer(self):
        return self._l_out

    @property
    def input_var(self):
        return self._l_in.input_var

    @property
    def target_var(self):
        return self._l_tar.input_var

    @property
    def layers(self):
        return self._layers

    @property
    def output(self):
        return self._output

    @property
    def n_params(self):
        return sum([np.prod(param.get_shape()).value for param in self.get_params()])


class DeterministicNetwork(NeuralNetwork):

    def predict(self, X):

        if self.output_layer.nonlinearity == tf.nn.softmax:
            y_p = tf.argmax(self._output,1)
        else:
            y_p = self._output

        Y_p = self._predict(y_p, X)
        return Y_p

class StochasticNetwork(NeuralNetwork):

    def predict(self, X, k= 1):
        sess = tf.get_default_session()

        o_p= []
        for _ in range(k):

            o_p.append(self._predict(self._output, X))
            o_p = np.concatenate([o[None,...] for o in o_p], axis= 0)
            mu_p = np.mean(o_p,axis= 0)
            std_p = np.std(o_p,axis= 0)

        if self.output_layer.nonlinearity == tf.nn.softmax:
            Y_p = np.argmax(mu_p,1)
        elif self.output_layer.nonlinearity == tf.identity:
            Y_p = mu_p
        elif self.output_layer.nonlinearity == tf.nn.sigmoid:
            Y_p = mu_p

        return Y_p


class MLP(LayersPowered, Serializable, DeterministicNetwork):
    def __init__(self, name, input_shape, output_dim, hidden_sizes, hidden_nonlinearity,
                 output_nonlinearity, hidden_W_init=L.XavierUniformInitializer(), hidden_b_init=tf.zeros_initializer,
                 output_W_init=L.XavierUniformInitializer(), output_b_init=tf.zeros_initializer, batch_size=None,
                 input_var=None, input_layer=None, batch_normalization=False, weight_normalization=False,
                 ):

        Serializable.quick_init(self, locals())
        self.name= name

        with tf.variable_scope(name):
            if input_layer is None:
                l_in = L.InputLayer(shape=(batch_size,) + input_shape, input_var=input_var, name="input")
            else:
                l_in = input_layer
            self._layers = [l_in]
            l_hid = l_in
            if batch_normalization:
                ls = L.batch_norm(l_hid)
                l_hid = ls[-1]
                self._layers += ls
            for idx, hidden_size in enumerate(hidden_sizes):
                l_hid = L.DenseLayer(
                    l_hid,
                    num_units=hidden_size,
                    nonlinearity=hidden_nonlinearity,
                    name="hidden_%d" % idx,
                    W=hidden_W_init,
                    b=hidden_b_init,
                    weight_normalization=weight_normalization
                )
                if batch_normalization:
                    ls = L.batch_norm(l_hid)
                    l_hid = ls[-1]
                    self._layers += ls
                self._layers.append(l_hid)
            l_out = L.DenseLayer(
                l_hid,
                num_units=output_dim,
                nonlinearity=output_nonlinearity,
                name="output",
                W=output_W_init,
                b=output_b_init,
                weight_normalization=weight_normalization
            )
            if batch_normalization:
                ls = L.batch_norm(l_out)
                l_out = ls[-1]
                self._layers += ls
            self._layers.append(l_out)
            self._l_in = l_in
            self._l_out = l_out
            self._l_tar = L.InputLayer(shape=(batch_size,) + (output_dim,), input_var=input_var, name="target")

            # self._input_var = l_in.input_var
            self._output = L.get_output(l_out)

            LayersPowered.__init__(self, l_out)

def convolution(l_in, hws, channels, strides, pads, nonlinearity, linear_output=False, drop_prob=0.0):
    fan_in = l_in.get_shape()[-1]
    n_conv = len(hws)
    for j, (hw,stride,pad,fan_out) in enumerate(zip(hws,strides,pads,channels)):
        w = tf.get_variable("w_cnn{}".format(j), shape=(hw,hw,fan_in,fan_out), dtype=tf.float32,
                            initializer=tf.contrib.layers.xavier_initializer_conv2d())
        b = tf.get_variable("b_cnn{}".format(j), shape=(fan_out,), dtype=tf.float32,
                            initializer=tf.zeros_initializer)
        l_in = tf.nn.conv2d(l_in, w, [1,stride,stride,1], pad) + b
        if j != (n_conv - 1):
            l_in = tf.nn.dropout(l_in, 1.-drop_prob)
            l_in = nonlinearity(l_in)
        elif not linear_output:
            l_in = nonlinearity(l_in)
        fan_in = fan_out
    return l_in

def feedforward_on_list(l_in_list, scope, hidden_sizes, nonlinearity,
                        hidden_W_init, hidden_b_init,
                        linear_output = True,
                        drop_prob = 0.0):
    l_ins = []
    for i, l_in in enumerate(l_in_list):
        if i > 0:
            scope.reuse_variables()
        l_in = feedforward(l_in, hidden_sizes, nonlinearity, 
                          hidden_W_init, True, 
                          hidden_b_init, True,
                          drop_prob= drop_prob,
                          linear_output= linear_output)
        l_ins.append(l_in)
    return l_ins

class InverseDynamicsModel(object):
    def __init__(self, name, trainer, obs_shape, act_dim, hidden_sizes, hidden_nonlinearity,
            active_epoch = 0, batch_size = None):
        self.x = tf.placeholder(tf.float32, shape=(batch_size,obs_shape),name="x")
        self.x_prime = tf.placeholder(tf.float32, shape=(batch_size,obs_shape),name="x_prime")
        self.a = tf.placeholder(tf.float32, shape=(batch_size,act_dim),name="a")
        self.trainer = trainer
        self.batch_size = batch_size
        self.active_epoch = active_epoch
        
        l_in = tf.concat(1,[self.x, self.x_prime])
        with tf.variable_scope(name):
            self.model = model = feedforward(l_in, hidden_sizes + [act_dim], hidden_nonlinearity,
                    L.XavierUniformInitializer(), True,
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

    def train(self, X, X_prime, A, disc_step):
        sess = tf.get_default_session()
        n_ = X.shape[0]
        n = X_prime.shape[0]
        n__ = A.shape[0]
        batch_size = self.batch_size
        if batch_size is None:
            batch_size = 10
        assert (n == n_) and (n == n__)
        for i in range(disc_step):
            ixs = np.random.permutation(range(0,n,batch_size))
            losses = []
            for j in ixs:
                X_batch = X[j:j+batch_size]
                X_prime_batch = X_prime[j:j+batch_size]
                A_batch = A[j:j+batch_size]
                
                _, loss_disc = sess.run([self.opt,self.cost],{self.x:X_batch,self.x_prime:X_prime_batch,self.a:A_batch})
                losses.append(loss_disc)
        l_d = np.mean(losses)
        return l_d

class AdaptiveRewardLSTM(object):
    def __init__(self, name, trainer, cell, max_steps, d_obs_shape, d_act_shape, t_obs_shape, t_act_shape,
                 d_hidden_sizes, t_hidden_sizes, c_hidden_sizes, hidden_nonlinearity, transform_actions = False,
                 t_w_trainable = True, t_b_trainable = True, d_trainable = True, cost_weight = 0.5,
                 hidden_W_init=L.XavierUniformInitializer(), hidden_b_init=tf.zeros_initializer,
                 output_W_init=L.XavierUniformInitializer(), output_b_init=tf.zeros_initializer, batch_size=None,
                 input_var=None, input_layer=None, batch_normalization=False, weight_normalization=False,
                 disable_policy=0, disable_flip_gradient=0, flip_reward=0,
                 d_drop_prob=0.0,c_drop_prob=0.0,t_drop_prob=0.0,wgan=False,share_weights=False,
                 clip_weights={},
                 conv_params={}):
        assert cost_weight >= 0.0 and cost_weight <= 1.0
        
        self.cell = None
        self.batch_size = batch_size = None
        self.wgan = wgan
        if wgan:
            print("USING WASSERSTEIN GAN")

        t_obs_shape_flat = np.prod(t_obs_shape)
        d_obs_shape_flat = np.prod(d_obs_shape)
        assert (not share_weights) or (t_obs_shape_flat == d_obs_shape_flat)        

        self.x_source = x_source = tf.placeholder(tf.float32, shape=(batch_size,max_steps,d_obs_shape_flat), name="x_source")
        self.x_target = x_target = tf.placeholder(tf.float32, shape=(batch_size,max_steps,t_obs_shape_flat), name="x_target")
        
        self.x_source_a = x_source_a = tf.placeholder(tf.float32, shape=(batch_size,max_steps,d_act_shape), name= "x_source_a")
        self.x_target_a = x_target_a = tf.placeholder(tf.float32, shape=(batch_size,max_steps,t_act_shape), name= "x_target_a")
        if transform_actions:
            assert len(d_obs_shape) <= 1
            assert len(t_obs_shape) <= 1
            x_source = tf.concat(2,[self.x_source, self.x_source_a])
            x_target = tf.concat(2,[self.x_target, self.x_target_a])
            d_obs_shape = tuple(d_obs_shape + d_act_shape)
            t_obs_shape = tuple(t_obs_shape + t_act_shape)
        if disable_flip_gradient:
            flip_g = lambda x : x
        else:
            flip_g = lambda x : flip_gradient(x)
        
        self.transf_outputs = {}
        
        self.transf_features = {}
        self.rnn_features = {}
        
        self.conf_logits = {}
        self.disc_logits = {}
        
        self.disc_outputs = {}
        self.conf_outputs = {}
        
        self.lr = lr = tf.placeholder(tf.float32,shape=())

        x_source_list = [tf.reshape(i, (-1,)+d_obs_shape) for i in tf.split(1, max_steps, x_source)]
        x_target_list = [tf.reshape(i, (-1,)+t_obs_shape) for i in tf.split(1, max_steps, x_target)]
		
        if transform_actions:
            x_source_a_list = [None] * len(x_source_list)
            x_target_a_list = [None] * len(x_target_list)
        else:
            x_source_a_list = [tf.reshape(i, (-1, d_act_shape)) for i in tf.split(1, max_steps, x_source_a)]
            x_target_a_list = [tf.reshape(i, (-1, t_act_shape)) for i in tf.split(1, max_steps, x_target_a)]

        # define transformers
        with tf.variable_scope("reward"):
            for name, list_in, list_a, obs_shape in [("source",x_source_list,x_source_a_list,d_obs_shape),
                                     ("target",x_target_list,x_target_a_list,t_obs_shape)]:
                scope_name = "transf"
                if not share_weights:
                    scope_name += "/{}".format(name)
                with tf.variable_scope(scope_name) as scope:
                    if share_weights and name == "target" : scope.reuse_variables()
                    l_ins, feats = [], []
                    for (i,l_in,x_a) in zip(range(max_steps),list_in,list_a):
                        # reshape input variable
                        if batch_size is None:
                            bs = -1
                        else:
                            bs = batch_size
                        l_in = tf.reshape(l_in, (bs,) + obs_shape)
                        # reuse scope
                        if i > 0:
                            scope.reuse_variables()
			# perform convolutions, if appropriate.
                        if len(obs_shape) > 1:
                            l_in = convolution(l_in, conv_params['hws'], conv_params['channels'], 
                                              conv_params['strides'], 
                                              conv_params['pads'], 
                                              hidden_nonlinearity, 
                                              linear_output=False, 
                                              drop_prob=0.0)
                            hidden_dim = np.prod(l_in.get_shape()[1:]).value
                            print("CONV OUTPUTS ARE {}-dimensional".format(hidden_dim))
                            l_in = tf.reshape(l_in, (-1,hidden_dim))
                        l_in = feedforward(l_in, t_hidden_sizes, hidden_nonlinearity, 
                                          hidden_W_init, t_w_trainable, 
                                          hidden_b_init, t_b_trainable,
                                          drop_prob= t_drop_prob,
                                          linear_output=False)
                        feats.append(l_in)
                        if not transform_actions:
                            l_in = tf.concat(1,[l_in,x_a])
                        l_ins.append(l_in)
                    self.transf_outputs[name] = l_ins
                    self.transf_features[name] = tf.pack(feats)
                        
            with tf.variable_scope("conf") as scope:
                for name in ["source","target"]:
                    if name == "target" : scope.reuse_variables()
                    l_ins = feedforward_on_list(map(flip_g, self.transf_outputs[name]),
                                                scope, c_hidden_sizes, hidden_nonlinearity,
                                                hidden_W_init, hidden_b_init)
                    self.conf_outputs[name] = l_ins
                    self.conf_logits[name] = tf.reshape(tf.pack(l_ins),(-1,1))

            # define graph for discriminator
            with tf.variable_scope("disc") as scope:            
                if cell is None:
                    for name in ["source","target"]:
                        if name == "target" : scope.reuse_variables()                    
                        l_ins = feedforward_on_list(self.transf_outputs[name], scope,
                                                    d_hidden_sizes, hidden_nonlinearity,
                                                    hidden_W_init, hidden_b_init)
                        self.disc_outputs[name] = tf.reshape(tf.pack(l_ins),(-1,1))
                        self.disc_logits[name] = tf.reshape(tf.pack(l_ins),(-1,1))
                else:
                    #cell = tf.nn.rnn_cell.BasicLSTMCell(lstm_units, state_is_tuple= True)
                    for i, name in enumerate(["source","target"]):
                        if i > 0:
                            scope.reuse_variables()
                        rnn_outputs, states = tf.nn.rnn(cell, self.transf_outputs[name],
                                                        #initial_state= initial_state,
                                                        dtype=tf.float32)
                        rnn_outputs = tf.pack(rnn_outputs)
                        rnn_outputs = tf.transpose(rnn_outputs, [1,0,2])
                        batch_size = tf.shape(rnn_outputs)[0]                    
                        stacked_rnn_outputs = tf.reshape(rnn_outputs, (batch_size * max_steps,cell._num_units))
                        self.rnn_features[name] = tf.pack(rnn_outputs)
                        
                        pred = feedforward(stacked_rnn_outputs, d_hidden_sizes, hidden_nonlinearity, 
                                          hidden_W_init, t_w_trainable, 
                                          hidden_b_init, t_b_trainable,
                                          drop_prob= d_drop_prob)
                        self.disc_outputs[name] = tf.reshape(pred,(batch_size,max_steps,1))
                        # WARNING: using [-1] here might break when eos != max_traj_len                            
                        self.disc_logits[name] = self.disc_outputs[name][:,-1,:]

        if flip_reward or wgan:
            self.rewards = self.disc_outputs["target"]
        else:
            self.rewards = -tf.log(1.0 - tf.nn.sigmoid(self.disc_outputs["target"])) * (1.-disable_policy)

        # cost on "reward" classifier
        r_logits = tf.concat(0,[self.disc_logits["target"],self.disc_logits["source"]])
        r_labels = tf.concat(0,[tf.zeros_like(self.disc_logits["target"]), tf.ones_like(self.disc_logits["source"])])
        if wgan:
            c_pi = tf.reduce_sum(self.disc_outputs["target"])
            c_ex = tf.reduce_sum(self.disc_outputs["source"])
            self.cost_disc = c_pi - c_ex
        else:
            self.cost_disc = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(r_logits,r_labels))
        # cost on domain "confusion" classifier
        c_logits = tf.concat(0,[self.conf_logits["target"],self.conf_logits["source"]])
        c_labels = tf.concat(0,[tf.zeros_like(self.conf_logits["target"]),tf.ones_like(self.conf_logits["source"])])
        self.cost_conf = tf.constant(cost_weight) * tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(c_logits,c_labels))
        
        #self.cost = self.cost_disc + self.cost_conf

        self.trainer = trainer(learning_rate=self.lr)
        self.opt_conf = self.trainer.minimize(self.cost_conf)
        self.opt_disc = self.trainer.minimize(self.cost_disc)
        
        self.clip_ops = []
        if self.wgan:
            for scope, value in clip_weights.items():
                vrs = [v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if v.name.split('/')[0] == "reward" and v.name.split('/')[1] == scope]
                self.clip_ops += [v.assign(tf.clip_by_value(v, -value, value)) for v in vrs]
        
        #self.grads_and_mags = {cost : self.get_grad_mags(cost) for cost in [self.cost, self.cost_disc, self.cost_conf]}        
        
    def get_grad_mags(self, cost):
        var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope="reward")
        grads_and_vars = self.trainer.compute_gradients(cost, var_list)
        grad_mags = {v.name : tf.nn.l2_loss(g) for g, v in grads_and_vars if g is not None}
        return grad_mags
        
    def train(self, obs_ex, act_ex, obs_pi, act_pi, working_lr, disc_step, batch_size=10):
        sess = tf.get_default_session()
        n_ = obs_ex.shape[0]
        n = obs_pi.shape[0]
        assert n == n_
        
        _, loss_conf = sess.run([self.opt_conf, self.cost_conf],
                                {self.x_source: obs_ex,
                                 self.x_source_a: act_ex,
                                 self.x_target: obs_pi,
                                 self.x_target_a: act_pi,
                                 self.lr: working_lr})        

        q = []
        disc_converged = False
        for i in range(disc_step):
            ixs = np.random.permutation(range(0,n,batch_size))
	    losses = []
            for j in ixs:
                obs_ex_batch = obs_ex[j:j+batch_size]
                act_ex_batch = act_ex[j:j+batch_size]
                obs_pi_batch = obs_pi[j:j+batch_size]
                act_pi_batch = act_pi[j:j+batch_size]                
                
                _, loss_disc = sess.run([self.opt_disc, self.cost_disc],
                         {self.x_source: obs_ex_batch,
                            self.x_source_a: act_ex_batch,
                            self.x_target: obs_pi_batch,
                            self.x_target_a: act_pi_batch,
                            self.lr: working_lr})
                _ = sess.run(self.clip_ops)
		losses.append(loss_disc)
	    l_d = np.mean(losses)
	    if len(q) == 5:
		    disc_converged = abs(l_d - np.mean(q)) < 1e-5
		    q = q[1:]
	    assert len(q) <= 5
	    q.append(l_d)
	    if disc_converged:
                break
        
        return l_d, loss_conf, disc_converged, i
    
    def _predict(self, fetches, feed):
        sess = tf.get_default_session()
        if self.batch_size is None:
            Y_p = sess.run(fetches,feed)
        else:
            feed = {k : np.repeat(v, self.batch_size, axis= 0) for k, v in feed.items()}
            Y_p = sess.run(fetches,feed)
            if type(Y_p) is list:
                Y_p = np.squeeze(np.array(Y_p))
            Y_p = Y_p[:,0]

        return Y_p
    
    def compute_features(self, X, X_a,rnn=False,label=0):
        if rnn:
            _f = self.rnn_features
        else:
            _f = self.transf_features

        if label == 0:
            _features = _f["target"]
            x = self.x_target
            x_a = self.x_target_a

        else:
            _features = _f["source"]
            x = self.x_source
            x_a = self.x_source_a

        features = self._predict(_features, {x: X, x_a: X_a})
        if type(features) is list:
            features = np.squeeze(np.array(features))        
        return features
    
    def compute_score(self, X, X_a,rnn=False,label=0):
        if rnn:
            _l = self.disc_logits
        else:
            _l = self.conf_logits
            
        if label == 0:
            logits = _l["target"]
            x = self.x_target
            x_a = self.x_target_a

        else:
            logits = _l["source"]
            x = self.x_source
            x_a = self.x_source_a

        score = self._predict(logits, {x: X, x_a: X_a})
        if type(score) is list:
            score = np.squeeze(np.array(score))        
        return score       

    def compute_reward(self, X, X_a):
        """
        compute surrogate reward for actions in the target domain.
        """
        if self.cell is None:
            X = np.expand_dims(X,1)
            X_a = np.expand_dims(X_a,1)
        else:
            X = X[None,...]
            X_a = X_a[None,...]
        Y_p = self._predict(self.rewards, {self.x_target: X,
                                      self.x_target_a: X_a})
        assert not np.isnan(np.sum(Y_p))
        if type(Y_p) is list:
            Y_p = np.squeeze(np.array(Y_p))
        return Y_p


class BaselineMLP(MLP, Baseline):
    def initialize_optimizer(self):
        self._optimizer = LbfgsOptimizer('optim')

        optimizer_args = dict(
            loss=self.loss(),
            target=self,
            inputs = [self.input_var, self.target_var],
            network_outputs=[self.output]
        )

        self._optimizer.update_opt(**optimizer_args)

    @overrides
    def predict(self, path):
        # X = np.column_stack((path['observations'], path['actions']))
        X = path['observations']
        return super(BaselineMLP, self).predict(X)

    @overrides
    def fit(self, paths):
        observations = np.concatenate([p["observations"] for p in paths])
        returns = np.concatenate([p["returns"] for p in paths])
        #self._regressor.fit(observations, returns.reshape((-1, 1)))
        self._optimizer.optimize([observations, returns[...,None]])

class ConvNetwork(LayersPowered, Serializable, DeterministicNetwork):
    def __init__(self, name, input_shape, output_dim,
                 conv_filters, conv_filter_sizes, conv_strides, conv_pads,
                 hidden_sizes, hidden_nonlinearity, output_nonlinearity,
                 hidden_W_init=L.XavierUniformInitializer(), hidden_b_init=tf.zeros_initializer,
                 output_W_init=L.XavierUniformInitializer(), output_b_init=tf.zeros_initializer,
                 input_var=None, input_layer=None, batch_normalization=False, weight_normalization=False):
        Serializable.quick_init(self, locals())
        """
        A network composed of several convolution layers followed by some fc layers.
        input_shape: (width,height,channel)
            HOWEVER, network inputs are assumed flattened. This network will first unflatten the inputs and then apply the standard convolutions and so on.
        conv_filters: a list of numbers of convolution kernel
        conv_filter_sizes: a list of sizes (int) of the convolution kernels
        conv_strides: a list of strides (int) of the conv kernels
        conv_pads: a list of pad formats (either 'SAME' or 'VALID')
        hidden_nonlinearity: a nonlinearity from tf.nn, shared by all conv and fc layers
        hidden_sizes: a list of numbers of hidden units for all fc layers
        """
        with tf.variable_scope(name):
            if input_layer is not None:
                l_in = input_layer
                l_hid = l_in
            elif len(input_shape) == 3:
                l_in = L.InputLayer(shape=(None, np.prod(input_shape)), input_var=input_var, name="input")
                l_hid = L.reshape(l_in, ([0],) + input_shape, name="reshape_input")
            elif len(input_shape) == 2:
                l_in = L.InputLayer(shape=(None, np.prod(input_shape)), input_var=input_var, name="input")
                input_shape = (1,) + input_shape
                l_hid = L.reshape(l_in, ([0],) + input_shape, name="reshape_input")
            else:
                l_in = L.InputLayer(shape=(None,) + input_shape, input_var=input_var, name="input")
                l_hid = l_in

            if batch_normalization:
                l_hid = L.batch_norm(l_hid)
            for idx, conv_filter, filter_size, stride, pad in zip(
                    range(len(conv_filters)),
                    conv_filters,
                    conv_filter_sizes,
                    conv_strides,
                    conv_pads,
            ):
                l_hid = L.Conv2DLayer(
                    l_hid,
                    num_filters=conv_filter,
                    filter_size=filter_size,
                    stride=(stride, stride),
                    pad=pad,
                    nonlinearity=hidden_nonlinearity,
                    name="conv_hidden_%d" % idx,
                    weight_normalization=weight_normalization,
                )
                if batch_normalization:
                    l_hid = L.batch_norm(l_hid)

            if output_nonlinearity == L.spatial_expected_softmax:
                assert len(hidden_sizes) == 0
                assert output_dim == conv_filters[-1] * 2
                l_hid.nonlinearity = tf.identity
                l_out = L.SpatialExpectedSoftmaxLayer(l_hid)
            else:
                l_hid = L.flatten(l_hid, name="conv_flatten")
                for idx, hidden_size in enumerate(hidden_sizes):
                    l_hid = L.DenseLayer(
                        l_hid,
                        num_units=hidden_size,
                        nonlinearity=hidden_nonlinearity,
                        name="hidden_%d" % idx,
                        W=hidden_W_init,
                        b=hidden_b_init,
                        weight_normalization=weight_normalization,
                    )
                    if batch_normalization:
                        l_hid = L.batch_norm(l_hid)
                l_out = L.DenseLayer(
                    l_hid,
                    num_units=output_dim,
                    nonlinearity=output_nonlinearity,
                    name="output",
                    W=output_W_init,
                    b=output_b_init,
                    weight_normalization=weight_normalization,
                )
                if batch_normalization:
                    l_out = L.batch_norm(l_out)
            self._l_in = l_in
            self._l_out = l_out
            # self._input_var = l_in.input_var

        LayersPowered.__init__(self, l_out)

    @property
    def input_layer(self):
        return self._l_in

    @property
    def output_layer(self):
        return self._l_out

    @property
    def input_var(self):
        return self._l_in.input_var

class LSTMNetwork(object):
    def __init__(self, name, input_shape, output_dim, hidden_dim, hidden_nonlinearity=tf.nn.relu,
                 lstm_layer_cls=L.LSTMLayer,
                 output_nonlinearity=None, input_var=None, input_layer=None, forget_bias=1.0, use_peepholes=False,
                 layer_args=None):
        with tf.variable_scope(name):
            if input_layer is None:
                l_in = L.InputLayer(shape=(None, None) + input_shape, input_var=input_var, name="input")
            else:
                l_in = input_layer
            l_step_input = L.InputLayer(shape=(None,) + input_shape, name="step_input")
            # contains previous hidden and cell state
            l_step_prev_state = L.InputLayer(shape=(None, hidden_dim * 2), name="step_prev_state")
            if layer_args is None:
                layer_args = dict()
            l_lstm = lstm_layer_cls(l_in, num_units=hidden_dim, hidden_nonlinearity=hidden_nonlinearity,
                                    hidden_init_trainable=False, name="lstm", forget_bias=forget_bias,
                                    cell_init_trainable=False, use_peepholes=use_peepholes, **layer_args)
            l_lstm_flat = L.ReshapeLayer(
                l_lstm, shape=(-1, hidden_dim),
                name="lstm_flat"
            )
            l_output_flat = L.DenseLayer(
                l_lstm_flat,
                num_units=output_dim,
                nonlinearity=output_nonlinearity,
                name="output_flat"
            )
            l_output = L.OpLayer(
                l_output_flat,
                op=lambda flat_output, l_input:
                tf.reshape(flat_output, tf.stack((tf.shape(l_input)[0], tf.shape(l_input)[1], -1))),
                shape_op=lambda flat_output_shape, l_input_shape:
                (l_input_shape[0], l_input_shape[1], flat_output_shape[-1]),
                extras=[l_in],
                name="output"
            )
            l_step_state = l_lstm.get_step_layer(l_step_input, l_step_prev_state, name="step_state")
            l_step_hidden = L.SliceLayer(l_step_state, indices=slice(hidden_dim), name="step_hidden")
            l_step_cell = L.SliceLayer(l_step_state, indices=slice(hidden_dim, None), name="step_cell")
            l_step_output = L.DenseLayer(
                l_step_hidden,
                num_units=output_dim,
                nonlinearity=output_nonlinearity,
                W=l_output_flat.W,
                b=l_output_flat.b,
                name="step_output"
            )

            self._l_in = l_in
            self._hid_init_param = l_lstm.h0
            self._cell_init_param = l_lstm.c0
            self._l_lstm = l_lstm
            self._l_out = l_output
            self._l_step_input = l_step_input
            self._l_step_prev_state = l_step_prev_state
            self._l_step_hidden = l_step_hidden
            self._l_step_cell = l_step_cell
            self._l_step_state = l_step_state
            self._l_step_output = l_step_output
            self._hidden_dim = hidden_dim

    @property
    def state_dim(self):
        return self._hidden_dim * 2

    @property
    def input_layer(self):
        return self._l_in

    @property
    def input_var(self):
        return self._l_in.input_var

    @property
    def output_layer(self):
        return self._l_out

    @property
    def recurrent_layer(self):
        return self._l_lstm

    @property
    def step_input_layer(self):
        return self._l_step_input

    @property
    def step_prev_state_layer(self):
        return self._l_step_prev_state

    @property
    def step_hidden_layer(self):
        return self._l_step_hidden

    @property
    def step_state_layer(self):
        return self._l_step_state

    @property
    def step_cell_layer(self):
        return self._l_step_cell

    @property
    def step_output_layer(self):
        return self._l_step_output

    @property
    def hid_init_param(self):
        return self._hid_init_param

    @property
    def cell_init_param(self):
        return self._cell_init_param

    @property
    def state_init_param(self):
        return tf.concat(axis=0, values=[self._hid_init_param, self._cell_init_param])


class GRUNetwork(object):
    def __init__(self, name, input_shape, output_dim, hidden_dim, hidden_nonlinearity=tf.nn.relu,
                 gru_layer_cls=L.GRULayer,
                 output_nonlinearity=None, input_var=None, input_layer=None, layer_args=None):
        with tf.variable_scope(name):
            if input_layer is None:
                l_in = L.InputLayer(shape=(None, None) + input_shape, input_var=input_var, name="input")
            else:
                l_in = input_layer
            l_step_input = L.InputLayer(shape=(None,) + input_shape, name="step_input")
            l_step_prev_state = L.InputLayer(shape=(None, hidden_dim), name="step_prev_state")
            if layer_args is None:
                layer_args = dict()
            l_gru = gru_layer_cls(l_in, num_units=hidden_dim, hidden_nonlinearity=hidden_nonlinearity,
                                  hidden_init_trainable=False, name="gru", **layer_args)
            l_gru_flat = L.ReshapeLayer(
                l_gru, shape=(-1, hidden_dim),
                name="gru_flat"
            )
            l_output_flat = L.DenseLayer(
                l_gru_flat,
                num_units=output_dim,
                nonlinearity=output_nonlinearity,
                name="output_flat"
            )
            l_output = L.OpLayer(
                l_output_flat,
                op=lambda flat_output, l_input:
                tf.reshape(flat_output, tf.pack((tf.shape(l_input)[0], tf.shape(l_input)[1], -1))),
                shape_op=lambda flat_output_shape, l_input_shape:
                (l_input_shape[0], l_input_shape[1], flat_output_shape[-1]),
                extras=[l_in],
                name="output"
            )
            l_step_state = l_gru.get_step_layer(l_step_input, l_step_prev_state, name="step_state")
            l_step_hidden = l_step_state
            l_step_output = L.DenseLayer(
                l_step_hidden,
                num_units=output_dim,
                nonlinearity=output_nonlinearity,
                W=l_output_flat.W,
                b=l_output_flat.b,
                name="step_output"
            )

            self._l_in = l_in
            self._hid_init_param = l_gru.h0
            self._l_gru = l_gru
            self._l_out = l_output
            self._l_step_input = l_step_input
            self._l_step_prev_state = l_step_prev_state
            self._l_step_hidden = l_step_hidden
            self._l_step_state = l_step_state
            self._l_step_output = l_step_output
            self._hidden_dim = hidden_dim

    @property
    def state_dim(self):
        return self._hidden_dim

    @property
    def hidden_dim(self):
        return self._hidden_dim

    @property
    def input_layer(self):
        return self._l_in

    @property
    def input_var(self):
        return self._l_in.input_var

    @property
    def output_layer(self):
        return self._l_out

    @property
    def recurrent_layer(self):
        return self._l_gru

    @property
    def step_input_layer(self):
        return self._l_step_input

    @property
    def step_prev_state_layer(self):
        return self._l_step_prev_state

    @property
    def step_hidden_layer(self):
        return self._l_step_hidden

    @property
    def step_state_layer(self):
        return self._l_step_state

    @property
    def step_output_layer(self):
        return self._l_step_output

    @property
    def hid_init_param(self):
        return self._hid_init_param

    @property
    def state_init_param(self):
        return self._hid_init_param

