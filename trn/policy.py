import tensorflow as tf
import numpy as np

import tf_rllab.core.layers as L
from tf_rllab.core.network import NeuralNetwork, DeterministicNetwork
from tf_rllab.core.layers_powered import LayersPowered
from rllab.core.serializable import Serializable

from tf_rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.misc.overrides import overrides

def feedforward(l_hid, hidden_sizes, hidden_nonlinearity,
        weight_normalization=False,
        hidden_W_init=L.XavierUniformInitializer(),
        hidden_b_init=tf.zeros_initializer,
        linear_output = False,
        start_idx = 0):
    for idx, hidden_size in enumerate(hidden_sizes):
        if linear_output and (idx == (len(hidden_sizes) - 1)):
            nonlin = None
        else:
            nonlin = hidden_nonlinearity
        l_hid = L.DenseLayer(
            l_hid,
            num_units=hidden_size,
            nonlinearity=hidden_nonlinearity,
            name="hidden_%d" % (idx + start_idx),
            W=hidden_W_init,
            b=hidden_b_init,
            weight_normalization=weight_normalization
        )
    return l_hid

def kl_from_prior(mu, sig, z_dim):
    #sig = tf.exp(self.z_logstd) + 1e-3
    el_1 = -0.5 * tf.to_float(z_dim)
    el_2 = -tf.reduce_sum(tf.log(sig), 1)
    el_3 = 0.5*tf.reduce_sum(tf.square(sig), 1)
    el_4 = 0.5*tf.reduce_sum(tf.square(mu), 1)
    return el_1 + el_2 + el_3 + el_4

class LMLP(LayersPowered, Serializable, NeuralNetwork):
    """
    MLP policy with "latent" (variational) layer. Written using rllab, so can
    act as a policy.
    """
    def __init__(self, name, input_shape, output_dim, z_dim,
                 pre_hidden_sizes, post_hidden_sizes, hidden_nonlinearity,
                 output_nonlinearity, hidden_W_init=L.XavierUniformInitializer(), hidden_b_init=tf.zeros_initializer,
                 output_W_init=L.XavierUniformInitializer(), output_b_init=tf.zeros_initializer, batch_size=None,
                 input_var=None, input_layer=None, weight_normalization=False):

        Serializable.quick_init(self, locals())
        self.name= name

        with tf.variable_scope(name):
            if input_layer is None:
                l_in = L.InputLayer(shape=(batch_size,) + input_shape, input_var=input_var, name="input")
            else:
                l_in = input_layer
            self._layers = [l_in]

            # construct graph
            l_hid = feedforward(l_in,pre_hidden_sizes,hidden_nonlinearity,
                                hidden_W_init=hidden_W_init,
                                hidden_b_init=hidden_b_init,
                                weight_normalization=weight_normalization,
                                start_idx = 0)
            l_lat = L.LatentLayer(l_hid, z_dim)
            l_hid = feedforward(l_lat,post_hidden_sizes,hidden_nonlinearity,
                                hidden_W_init=hidden_W_init,
                                hidden_b_init=hidden_b_init,
                                weight_normalization=weight_normalization,
                                start_idx = len(pre_hidden_sizes))

            # create output layer
            l_out = L.DenseLayer(
                l_hid,
                num_units=output_dim,
                nonlinearity=output_nonlinearity,
                name="output",
                W=output_W_init,
                b=output_b_init,
                weight_normalization=weight_normalization
            )

            self._layers.append(l_out)
            self._l_lat = l_lat
            self._z_dim = z_dim
            self._l_in = l_in
            self._l_out = l_out
            self._l_tar = L.InputLayer(shape=(batch_size,) + (output_dim,), input_var=input_var, name="target")

            # complexity loss for variational posterior
            z_mu, z_sig = self._l_lat.get_dparams_for(L.get_output(self._l_lat.input_layer))
            self.kl_cost = kl_from_prior(z_mu, z_sig, self._z_dim)

            # self._input_var = l_in.input_var
            self._output = L.get_output(l_out)

            LayersPowered.__init__(self, l_out)

    def compute_kl(self, X):
        """
        Computes the KL divergence between variational posterior and prior.
        """
        sess = tf.get_default_session()
        return sess.run(self.kl_cost, {self._l_in.input_var : X})


class MergeMLP(LayersPowered, Serializable, DeterministicNetwork):
    def __init__(self, name, input_shape, output_dim, hidden_sizes,
                 hidden_nonlinearity, output_nonlinearity,
                 z_dim, z_idx, z_hidden_sizes, merge="mul",
                 hidden_W_init=L.XavierUniformInitializer(), hidden_b_init=tf.zeros_initializer,
                 output_W_init=L.XavierUniformInitializer(), output_b_init=tf.zeros_initializer, batch_size=None,
                 input_var=None, input_layer=None, batch_normalization=False, weight_normalization=False,
                 ):

        Serializable.quick_init(self, locals())
        self.name= name

        total_dim = np.prod(input_shape)

        with tf.variable_scope(name):
            if input_layer is None:
                l_in = L.InputLayer(shape=(batch_size,) + input_shape, input_var=input_var, name="input")
            else:
                l_in = input_layer
            self._layers = [l_in]

            # slice off features / observation
            l_feat = L.SliceLayer(
                    l_in,
                    indices=slice(0, total_dim - z_dim),
                    name="l_feat")

            # slice off z "style" variable
            l_z = L.SliceLayer(
                    l_in,
                    indices=slice(total_dim - z_dim, total_dim),
                    name="l_z")

            l_pre = feedforward(l_feat, hidden_sizes[:z_idx], hidden_nonlinearity,
                    linear_output = True)
            with tf.variable_scope("z"):
                # if merging mul, ensure dimensionalities match.
                if merge == "mul":
                    _head = [total_dim] + hidden_sizes
                    _head = [_head[z_idx]]
                elif merge == "concat":
                    _head = []
                l_z = feedforward(l_z, z_hidden_sizes + _head,
                        hidden_nonlinearity,
                        linear_output = True)

            # merge latent code with features
            if merge == "mul":
                l_merge = L.ElemwiseMulLayer([l_pre,l_z])
            elif merge == "concat":
                l_merge = L.ConcatLayer([l_pre, l_z],axis=1)
            else: raise NotImplementedError

            if z_idx > 0:
                l_merge = L.NonlinearityLayer(l_merge, hidden_nonlinearity)
            l_hid = feedforward(l_merge, hidden_sizes[z_idx:],
                    hidden_nonlinearity,
                    start_idx = z_idx)

            l_out = L.DenseLayer(
                l_hid,
                num_units=output_dim,
                nonlinearity=output_nonlinearity,
                name="output",
                W=output_W_init,
                b=output_b_init,
                weight_normalization=weight_normalization
            )
            #if batch_normalization:
            #    ls = L.batch_norm(l_out)
            #    l_out = ls[-1]
            #    self._layers += ls
            self._layers.append(l_out)
            self._l_in = l_in
            self._l_out = l_out
            self._l_tar = L.InputLayer(shape=(batch_size,) + (output_dim,), input_var=input_var, name="target")

            # self._input_var = l_in.input_var
            self._output = L.get_output(l_out)

            LayersPowered.__init__(self, l_out)

#class FactoredMLP(LayersPowered, Serializable, DeterministicNetwork):
#    def __init__(self, name, n_agents, input_shape, output_dim, hidden_sizes, hidden_nonlinearity,
#                 output_nonlinearity, hidden_W_init=L.XavierUniformInitializer(), hidden_b_init=tf.zeros_initializer,
#                 output_W_init=L.XavierUniformInitializer(), output_b_init=tf.zeros_initializer, batch_size=None,
#                 input_var=None, input_layer=None, batch_normalization=False, weight_normalization=False,
#                 ):
#
#        Serializable.quick_init(self, locals())
#        self.name= name
#
#        input_dim = np.prod(input_shape) / n_agents
#        with tf.variable_scope(name):
#            if input_layer is None:
#                l_in = L.InputLayer(shape=(batch_size, input_dim), input_var=input_var, name="input")
#            else:
#                l_in = input_layer
#            self._layers = [l_in]
#            l_hid = l_in
#
#            for idx, hidden_size in enumerate(hidden_sizes):
#
#                l_hid = L.DenseLayer(
#                    l_hid,
#                    num_units=hidden_size,
#                    nonlinearity=hidden_nonlinearity,
#                    name="hidden_%d" % idx,
#                    W=hidden_W_init,
#                    b=hidden_b_init,
#                    weight_normalization=weight_normalization,
#                    variable_reuse = bool(tower_ix > 0)
#                )
#
#            l_out = L.DenseLayer(
#                l_hid,
#                num_units=output_dim/n_agents,
#                nonlinearity=output_nonlinearity,
#                name="output",
#                W=output_W_init,
#                b=output_b_init,
#                weight_normalization=weight_normalization,
#                variable_reuse = bool(tower_ix > 0)
#            )
#
#            self._layers.append(l_out)
#            self._l_in = l_in
#            self._l_out = l_out
#            self._l_tar = L.InputLayer(shape=(batch_size,) + (output_dim,), input_var=input_var, name="target")
#
#            # self._input_var = l_in.input_var
#            self._output = L.get_output(l_out)
#
#            LayersPowered.__init__(self, l_out)

class FactoredMLPPolicy(GaussianMLPPolicy):
    @overrides
    def get_action(self, observation):
        flat_obs = self.observation_space.flatten(observation)
        n_agents = len(observation) / self.obs_dim

        actions = []
        means = []
        log_stds = []
        for i in range(n_agents):
            flat_ob = flat_obs[i*self.obs_dim : i*self.obs_dim + self.obs_dim]

            mean, log_std = [x[0] for x in self._f_dist([flat_ob])]
            rnd = np.random.normal(size=mean.shape)
            action = rnd * np.exp(log_std) + mean

            means.append(mean)
            log_stds.append(log_std)
            actions.append(action)

        action = np.concatenate(actions)
        mean = np.concatenate(means)
        log_std = np.concatenate(log_stds)

        return action, dict(mean=mean, log_std=log_std)

