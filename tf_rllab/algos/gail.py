from rllab.misc import ext
from rllab.misc.overrides import overrides

from tf_rllab.algos.trpo import TRPO
from tf_rllab.optimizers.first_order_optimizer import Solver, SimpleSolver

from rllab.misc.overrides import overrides
import rllab.misc.logger as logger
from tf_rllab.optimizers.penalty_lbfgs_optimizer import PenaltyLbfgsOptimizer
from tf_rllab.algos.batch_polopt import BatchPolopt
from tf_rllab.misc import tensor_utils

import tensorflow as tf
import numpy as np

class GAIL(TRPO):
    """
    Generative adversarial imitation learning.
    """

    def __init__(
            self,
	    debug = False,
            optimizer=None,
            optimizer_args=None,
            fo_optimizer_cls=None,
            fo_optimizer_args=None,
            reward=None,
	    transformer=None,
            invdyn=None,
            expert_data=None,
            gail_batch_size=100,
            disc_steps=1,
            decay_steps=1,
            decay_rate=1.0,
            act_mean = 0.0,
            act_std = 1.0,
            hard_freeze = True,
            freeze_upper = 1.0,
            freeze_lower = 0.5,
	        expert_obs_noise_scale = 0.0,
	        add_noise = False,
	        use_trajs = False,
            **kwargs):
		super(GAIL, self).__init__(optimizer=optimizer, optimizer_args=optimizer_args, **kwargs)
		self.debug = debug

		self.reward_model = reward
                self.invdyn_model = invdyn
		self.transformer_model = transformer
		self.expert_data = expert_data
		self.gail_batch_size = gail_batch_size

		self.act_mean = act_mean
		self.act_std = act_std

		self.background_lr = fo_optimizer_args['learning_rate']
		self.working_lr = fo_optimizer_args['learning_rate']
		self.decay_rate = decay_rate
		self.decay_steps = decay_steps
				
		self.disc_steps = disc_steps

		self.hard_freeze = hard_freeze
		self.freeze_upper = freeze_upper
		self.freeze_lower = freeze_lower
				
		self.expert_obs_noise_scale = expert_obs_noise_scale
		self.use_trajs = use_trajs

    @overrides
    def optimize_policy(self, itr, samples_data):
		"""
		Perform policy optimization with TRPO, then draw samples from paths
		to fit discriminator/surrogate reward network.
		"""

		all_input_values = tuple(ext.extract(
			samples_data,
			"observations", "actions", "advantages"
		))
		agent_infos = samples_data["agent_infos"]
		state_info_list = [agent_infos[k] for k in self.policy.state_info_keys]
		dist_info_list = [agent_infos[k] for k in self.policy.distribution.dist_info_keys]
		all_input_values += tuple(state_info_list) + tuple(dist_info_list)
		if self.policy.recurrent:
			all_input_values += (samples_data["valids"],)

		# update policy
		super(GAIL, self).optimize_policy_from_inputs(all_input_values)

		# update discriminator
		if False:
			obs_pi = all_input_values[0].reshape((-1, np.prod(self.env.observation_space.shape)))
			act_pi = all_input_values[1].reshape((-1, np.prod(self.env.action_space.shape)))
		else:
			obs_pi, act_pi = self.extract_samples_data(samples_data)

		obs_ex= self.expert_data['obs']
		act_ex= self.expert_data['act']
                if self.invdyn_model is not None:
                    # fit the dynamics model
                    X = obs_pi[:,:-1,:]
                    X_prime = obs_pi[:,1:,:]
                    A = act_pi[:,:-1,:]
                    b_x, t_x, f_x = X.shape
                    b_a, t_a, f_a = A.shape
                    X = np.reshape(X,(b_x * t_x, f_x))
                    X_prime = np.reshape(X_prime,(b_x*t_x,f_x))
                    A = np.reshape(A,(b_a*t_a,f_a))

                    invdyn_loss = self.invdyn_model.train(X, X_prime, A, 20)
                    logger.record_tabular("invdyn_loss",invdyn_loss)
                    if self.invdyn_model.active_epoch < itr:
                        act_ex = self.invdyn_model.predict(obs_ex)
			if self.debug:
			    print("HURTING INVDYN")
			    act_ex = np.zeros_like(act_ex)
                        obs_ex = obs_ex[:,:-1,:]
                        n0, _, _ = act_ex.shape
                        n1, _, _ = obs_ex.shape
                        assert n0 == n1

		if not self.use_trajs:
                    b_o, t_o, f_o = obs_ex.shape
                    b_a, t_a, f_a = act_ex.shape
                    obs_ex = np.reshape(obs_ex,(b_o*t_o,1,f_o))
                    act_ex = np.reshape(act_ex,(b_a*t_a,1,f_a))
                                            
                    b_o, t_o, f_o = obs_pi.shape
                    b_a, t_a, f_a = act_pi.shape
                    obs_pi = np.reshape(obs_pi,(b_o*t_o,1,f_o))
                    act_pi = np.reshape(act_pi,(b_a*t_a,1,f_a))			

		p_ex = np.random.choice(obs_ex.shape[0], size=(self.gail_batch_size,), replace=False)
		p_pi = np.random.choice(obs_pi.shape[0], size=(self.gail_batch_size,), replace=False)

		obs_pi_batch = obs_pi[p_pi]
		act_pi_batch = act_pi[p_pi]

		obs_ex_batch = obs_ex[p_ex]
		act_ex_batch = act_ex[p_ex]

		obs_ex_batch += self.expert_obs_noise_scale * \
			np.random.normal(size=obs_ex_batch.shape)

		loss_di, loss_de, d_converged, d_converged_itr = \
			self.reward_model.train(obs_ex_batch, act_ex_batch, obs_pi_batch, act_pi_batch,
				                    self.working_lr, self.disc_steps)

		features_pi = self.reward_model.compute_features(obs_pi_batch, act_pi_batch,label=0)
		features_ex = self.reward_model.compute_features(obs_ex_batch, act_ex_batch,label=1)
				
		if self.use_trajs:
			rnn_features_pi = self.reward_model.compute_features(obs_pi_batch, act_pi_batch,label=0,
							                                     rnn=True)
			rnn_features_ex = self.reward_model.compute_features(obs_ex_batch, act_ex_batch,label=1,
							                                     rnn=True)		

		for (name,rnn_on) in [("conf",False), ("disc",True)]:
			if name == "disc" and self.reward_model.wgan:
				logger.record_tabular('disc_converged',d_converged)
				logger.record_tabular('disc_converged_itr',d_converged_itr)
			else:
				scores_pi = self.reward_model.compute_score(obs_pi_batch, act_pi_batch,
									                        rnn=rnn_on,label=0)
				scores_ex = self.reward_model.compute_score(obs_ex_batch, act_ex_batch,
									                        rnn=rnn_on,label=1)
				scores = np.row_stack((scores_pi, scores_ex))
				labels = np.row_stack((np.zeros_like(scores_pi),
									   np.ones_like(scores_ex)))			
	
				accuracy = ((scores < 0) == (labels == 0)).mean()
				accuracy_for_currpolicy = (scores_pi <= 0).mean()
				accuracy_for_expert = (scores_ex  > 0).mean()
				assert np.allclose(accuracy, .5*(accuracy_for_currpolicy + accuracy_for_expert))
				logger.record_tabular(name+'_racc', accuracy)
				logger.record_tabular(name+'_raccpi', accuracy_for_currpolicy)
				logger.record_tabular(name+'_raccex', accuracy_for_expert)			

		self.working_lr = self.background_lr * np.power(self.decay_rate, itr/self.decay_steps)

		logger.record_tabular('working_lr', self.working_lr)
		logger.record_tabular('background_lr', self.background_lr)
		#logger.record_tabular('racc', accuracy)
		#logger.record_tabular('raccpi', accuracy_for_currpolicy)
		#logger.record_tabular('raccex', accuracy_for_expert)

		names = ["loss_di","loss_de","features_pi","features_ex"]
		data = [loss_di, loss_de, features_pi, features_ex]
		if self.use_trajs:
			names += ["rnn_features_pi","rnn_features_ex"]
			data += [rnn_features_pi, rnn_features_ex]
		self.policy.save_extra_data(names, data, itr=itr)
		return dict()
			
    def extract_samples_data(self,samples_data):
		"""
		This will need to be more complicated in order to support eos.
		"""
		obs_pi = np.array([path['observations'] for path in samples_data['paths']])
		act_pi = np.array([path['actions'] for path in samples_data['paths']])
		return obs_pi, act_pi

    @overrides
    def process_samples(self, itr, paths):
        path_lengths = []

        for path in paths:
			X = np.column_stack((path['observations'],path['actions']))
			
			path['env_rewards'] = path['rewards']
			rewards = np.squeeze(
				self.reward_model.compute_reward(path['observations'], path['actions'])
			)
			
			if rewards.ndim == 0:
				rewards = rewards[np.newaxis]
			path['rewards'] = rewards
						
			assert not np.isnan(np.sum(rewards))
			
			path_lengths.append(X.shape[0])

        assert all([path['rewards'].ndim == 1 for path in paths])
        logger.record_tabular('pathLengths',np.mean(path_lengths))
        return self.sampler.process_samples(itr, paths)

