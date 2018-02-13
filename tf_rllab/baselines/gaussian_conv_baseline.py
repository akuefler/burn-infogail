import tensorflow as tf
from rllab.baselines.base import Baseline


import numpy as np

from rllab.misc.overrides import overrides
from rllab.baselines.base import Baseline
from tf_rllab.regressors.gaussian_mlp_regressor import GaussianMLPRegressor

from tf_rllab.core.network import ConvNetwork
from tf_rllab.core.parameterized import Parameterized, Serializable

class GaussianConvBaseline(Baseline, Parameterized, Serializable):

    def __init__(
            self,
            env_spec,
            subsample_factor=1.,
            hidden_nonlinearity=tf.nn.relu,
            conv_filters=[32,32,64], 
            conv_filter_sizes=[8,6,4], 
            conv_strides=[4,3,2], 
            hidden_sizes=(32,32),
            regressor_args=None,
    ):
        Serializable.quick_init(self, locals())
        super(GaussianConvBaseline, self).__init__(env_spec)
        if regressor_args is None:
            regressor_args = dict()
            
        with tf.variable_scope("baseline"):
            obs_shape_flat = np.prod(env_spec.observation_space.shape)
            mean_network = ConvNetwork("cnn", env_spec.observation_space.shape, 1, 
                                      conv_filters, 
                                      conv_filter_sizes, 
                                      conv_strides, 
                                      ["VALID"]*len(conv_filters), 
                                      hidden_sizes, 
                                      hidden_nonlinearity, 
                                      output_nonlinearity=None)
            self._regressor = GaussianMLPRegressor("base", (obs_shape_flat,), 1, 
                                                  mean_network=mean_network)
        #self._regressor = GaussianConvRegressor(
            #input_shape=env_spec.observation_space.shape,
            #output_dim=1,
            #name="vf",
            #**regressor_args
        #)

    @overrides
    def fit(self, paths):
        observations = np.concatenate([p["observations"] for p in paths])
        returns = np.concatenate([p["returns"] for p in paths])
        self._regressor.fit(observations, returns.reshape((-1, 1)))

    @overrides
    def predict(self, path):
        return self._regressor.predict(path["observations"]).flatten()

    @overrides
    def get_param_values(self, **tags):
        return self._regressor.get_param_values(**tags)

    @overrides
    def set_param_values(self, flattened_params, **tags):
        self._regressor.set_param_values(flattened_params, **tags)