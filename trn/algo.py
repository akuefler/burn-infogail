from tf_rllab.algos.npo import NPO
from tf_rllab.algos.trpo import TRPO
from tf_rllab.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer

import numpy as np
import rllab.misc.logger as logger
from network import InverseDynamicsModel, RewardModel

from rllab.misc.overrides import overrides

from rllab.config import *
from trn.config import policy_paths

from rltools.util import nan_stack_ragged_array

import h5py

from scipy.stats import mode, entropy
import sklearn.metrics as skmets

from collections import Counter, defaultdict

from sklearn.metrics import auc

class ReplayBuffer(object):
    def __init__(self):
        self.B = []

    def add(self, paths):
        self.B += paths

    def sample(self, n):
        return np.random.choice(self.B, n)

class GAIL(TRPO):
    """
    Trust Region Policy Optimization
    """

    def __init__(
            self,
            exp_name,
            exp_id,
            expert_data,
            reward_model,
            gail_batch_size,
            invdyn_model=None,
            info_model=None,
            optimizer=None,
            optimizer_args=None,
            kl_weight = 1.0,
            rew_aug = 0.0,
            debug=False,
            normalize_actions=True,
            model_all = False,
            curriculum = {},
            use_replay_buffer = False,
            **kwargs):
        super(GAIL, self).__init__(optimizer=optimizer, optimizer_args=optimizer_args, **kwargs)
        self.exp_name = exp_name
        self.exp_id = exp_id

        self.expert_data = expert_data
        self.reward_model = reward_model
        self.invdyn_model = invdyn_model
        self.info_model = info_model
        self.gail_batch_size = gail_batch_size
        self.debug = debug
        self.max_path_length = kwargs["max_path_length"]
        self.model_all = model_all
        self.curriculum = curriculum
        self.use_replay_buffer = use_replay_buffer

        self.wrapped_env = kwargs['env'].wrapped_env
        self.a_mean = self.wrapped_env.act_mean
        self.a_std = self.wrapped_env.act_std
        self.rew_aug = rew_aug
        if not normalize_actions:
            self.a_mean = np.zeros_like(self.a_mean)
            self.a_std = np.ones_like(self.a_std)

        # vae param
        self.replay_buffer = ReplayBuffer()
        self.clustering_scores = dict(
                ami = skmets.adjusted_mutual_info_score,
                )

        self.records = defaultdict(list)

    def apply_curriculum(self, itr):
        if (itr + 1) % self.curriculum["step"] == 0:
            self.wrapped_env.n_egos = self.wrapped_env.n_egos + self.curriculum['add']
        logger.record_tabular("n_egos", self.wrapped_env.n_egos)

    def optimize_policy(self, itr, samples_data):
        # update policy
        super(GAIL, self).optimize_policy(itr, samples_data)

        # retrieve data matrices
        # in batch x timestep x feature format
        y = np.array([path['dom_ix'] for path in samples_data['paths']])
        x_pi = np.array([path['observations'] for path in samples_data['paths']])
        a_pi = np.array([path['actions'] for path in samples_data['paths']])
        x_ex = self.expert_data["train"]['obs']
        a_ex = self.expert_data["train"]['act']

        # InfoGAIL does not require "labels" for expert
        if self.info_model is not None:
            B, T, Do = x_ex.shape
            z_dim = self.info_model.z_dim

            z_ex = np.expand_dims(np.ones((B,z_dim)),axis=1).repeat(T,axis=1)*np.nan
            x_ex = np.concatenate([x_ex,z_ex],axis=-1)

        else:
            z_dim = 0

        assert np.allclose(x_ex.shape[:-1], a_ex.shape[:-1])

        # format as paddad BxTxD, for recurrent networks
        y, _ = nan_stack_ragged_array(y, self.max_path_length)
        x_ex, _ = nan_stack_ragged_array(x_ex, self.max_path_length)
        a_ex, _ = nan_stack_ragged_array(a_ex, self.max_path_length)
        x_pi, _ = nan_stack_ragged_array(x_pi, self.max_path_length)
        a_pi, l_pi = nan_stack_ragged_array(a_pi, self.max_path_length)

        # normalize policy actions (states normalized by environment)
        if self.info_model is not None or self.rew_aug < 1.0:
            a_pi = (a_pi - self.a_mean) / self.a_std

            if self.info_model is not None:
                self.update_info_model(itr, x_ex, a_ex, x_pi, a_pi, l_pi, y)

            x_ex, a_ex, x_pi, a_pi, y = self.stack_and_clean_trajs(x_ex, a_ex, x_pi, a_pi, y, z_dim = z_dim)

            gail_batch_size = np.minimum(x_pi.shape[0], self.gail_batch_size)

            # extract batches from expert and policy training sets.
            ix_ex = np.random.choice(np.arange(x_ex.shape[0]),gail_batch_size,replace=False)
            ix_pi = np.random.choice(np.arange(x_pi.shape[0]),gail_batch_size,replace=False)

            x_ex = x_ex[ix_ex]
            a_ex = a_ex[ix_ex]
            x_pi = x_pi[ix_pi]
            a_pi = a_pi[ix_pi]

            # log difference in expert / policy statistics
            mean_diff = lambda x, y : np.linalg.norm(x.mean(axis=0) - y.mean(axis=0))
            std_diff = lambda x, y : np.linalg.norm(x.std(axis=0) - y.std(axis=0))

            # compute differences in distributions for debugging
            logger.record_tabular("x_mean_diff", mean_diff(x_ex[:,:-z_dim],x_pi[:,:-z_dim]))
            logger.record_tabular("x_std_diff", std_diff(x_ex[:,:-z_dim],x_pi[:,:-z_dim]))
            logger.record_tabular("a_mean_diff", mean_diff(a_ex,a_pi))
            logger.record_tabular("a_std_diff", std_diff(a_ex,a_pi))
            logger.record_tabular("z_mean_diff",
                    mean_diff(x_ex[:,-z_dim:],x_pi[:,-z_dim:]))
            logger.record_tabular("z_std_diff",
                    std_diff(x_ex[:,-z_dim:],x_pi[:,-z_dim:]))

            # TRAIN REWARD MODEL
            if self.rew_aug < 1.0:
                loss = self.reward_model.train(x_ex, a_ex, x_pi, a_pi)
                logger.record_tabular("gan_loss",loss)

                # Logging for WGAN
                if not self.reward_model.wgan:
                    scores_ex = self.reward_model.compute_score(x_ex, a_ex)
                    scores_pi = self.reward_model.compute_score(x_pi, a_pi)
                    scores = np.concatenate([scores_ex, scores_pi])
                    labels = np.concatenate([np.ones_like(scores_ex), np.zeros_like(scores_pi)])

                    accuracy = ((scores < 0.0) == (labels == 0)).mean()
                    acc_ex = (scores_ex > 0.0).mean()
                    acc_pi = (scores_pi <= 0.0).mean()
                    accuracy_ = 0.5 * (acc_ex + acc_pi)
                    assert np.allclose(accuracy, accuracy_)
                    logger.record_tabular('disc_racc',accuracy)
                    logger.record_tabular('disc_raccpi',acc_pi)
                    logger.record_tabular('disc_raccex',acc_ex)

        if self.curriculum != {}:
            self.apply_curriculum(itr)

    def stack_and_clean_trajs(self, x_ex, a_ex, x_pi, a_pi, y, z_dim = 0):
        if z_dim is None: z_dim = 0
        # flatten out trajectories
        x_ex = np.row_stack(x_ex)
        a_ex = np.row_stack(a_ex)
        x_pi = np.row_stack(x_pi)
        a_pi = np.row_stack(a_pi)

        # remove any nan rows
        x_ex = x_ex[~np.isnan(x_ex[...,:-z_dim]).any(axis=1)]
        a_ex = a_ex[~np.isnan(a_ex[...,:-z_dim]).any(axis=1)]
        x_pi = x_pi[~np.isnan(x_pi).any(axis=1)]
        a_pi = a_pi[~np.isnan(a_pi).any(axis=1)]

        y = y[~np.isnan(y)]

        return x_ex, a_ex, x_pi, a_pi, y

    def update_info_model(self, itr, x_ex, a_ex, x_pi, a_pi, l_pi, y):
        z_dim = self.info_model.z_dim

        if self.info_model.is_recurrent:
            loss = self.info_model.train(itr, x_pi, a_pi, y, EOS=l_pi)
            _, (info_acc, dom_acc) = self.info_model.predict(x_pi, a_pi, Y=y, EOS=l_pi)

        # TODO : not even using x_ex and a_ex here ...
        x_ex, a_ex, x_pi, a_pi, y = \
                self.stack_and_clean_trajs(x_ex, a_ex, x_pi, a_pi, y, z_dim = z_dim)

        if not self.info_model.is_recurrent:
            loss = self.info_model.train(itr, x_pi, a_pi, y)
            _, (info_acc, dom_acc) = self.info_model.predict(x_pi, a_pi, Y=y)

        logger.record_tabular("info_loss", loss)
        logger.record_tabular("info_acc", info_acc)
        logger.record_tabular("dom_acc", dom_acc)

        self.compute_mutual_information()

    def compute_mutual_information(self):
        for dataname, expert_data in self.expert_data.items():
            obs_B_T_Do = expert_data['obs']
            act_B_T_Da = expert_data['act']
            len_B = np.minimum(expert_data['exlen_B'], self.max_path_length)

            # get labels on track dataset
            cls_B = expert_data.get('cls_B',None)
            dom_B = expert_data.get('dom_B',None)

            B, _, Do = obs_B_T_Do.shape
            B, _, Da = act_B_T_Da.shape

            # truncate trajectories
            T = self.max_path_length
            obs_B_T_Do = obs_B_T_Do[:,:T,:]
            # have actions been normalized ?
            act_B_T_Da = act_B_T_Da[:,:T,:]

            # does this require preprocessing?
            if not self.info_model.is_recurrent:
                obs_BT_Do = np.reshape(obs_B_T_Do, (B * T, Do))
                act_BT_Da = np.reshape(act_B_T_Da, (B * T, Da))

                obs_BT_Do = obs_BT_Do[~np.isnan(obs_BT_Do).any(axis=1)]
                act_BT_Da = act_BT_Da[~np.isnan(act_BT_Da).any(axis=1)]

                clus_BT, _ = self.info_model.predict(np.concatenate([obs_BT_Do,
                    np.zeros_like(obs_BT_Do)[:,:self.info_model.z_dim]], axis=-1),
                    act_BT_Da)
                clus_B_T = np.reshape(clus_BT, (B,T))
                clus_B, _ = np.squeeze(mode(clus_B_T,axis=1))

            if self.info_model.is_recurrent:
                clus_B, _ = self.info_model.predict(np.concatenate([obs_B_T_Do,
                    np.zeros_like(obs_B_T_Do)[:,:,:self.info_model.z_dim]], axis=-1),
                    act_B_T_Da, EOS = len_B)

            self.record_freqs_and_ent(clus_B, dataname+"_clus")
            self.record_mutual_info_scores(dataname,cls_B, dom_B, clus_B)

    def record_mutual_info_scores(self, name, cls_B, dom_B, clus_B):
        for score_name, score_f in self.clustering_scores.items():
            score_cls = score_f(cls_B, clus_B)
            score_dom = score_f(dom_B, clus_B)

            logger.record_tabular('{}_{}_cls'.format(name,score_name), score_cls)
            logger.record_tabular('{}_{}_dom'.format(name,score_name), score_dom)

            # record data
            self.records['{}_{}_cls'.format(name,score_name)].append(score_cls)
            self.records['{}_{}_dom'.format(name,score_name)].append(score_dom)

            for dom in np.unique(dom_B):
                score_cls_in_dom = score_f(
                        cls_B[dom_B == dom], clus_B[dom_B ==dom]
                        )
                logger.record_tabular('{}_{}_cls_in_dom_{}'.format(name,score_name, dom),
                        score_cls_in_dom)


    def record_freqs_and_ent(self, preds, name):
        freqs = Counter(preds)
        pks = []
        for i in range(self.info_model.z_dim):
            pk = freqs[i] / np.float(np.sum(freqs.values()))
            logger.record_tabular('{}_freq_{}'.format(name,i), pk)
            pks.append(pk)
        logger.record_tabular('{}_entropy'.format(name), entropy(pks))

    def compare_results(self, itr):
        return False
#        #logger.record_tabular('',)
#        n = len(self.records["valid_ami_cls"])
#        ami = self.records['valid_ami_cls']
#        pl = self.records['path_lengths']
#
#        # normalize pl in 0-1 range
#        pl = np.array(pl) / self.max_path_length
#        score = auc(np.arange(n), np.maximum(0,np.mean([ami, pl], axis=0)))
#        logger.record_tabular('auc', score)
#
#        endtrain = False
#        i = (itr + 1) % 5
#        if i == 0 and itr > 23:
#            # open experiment log file
#            filename = "{}/logs/{}.txt".format(PROJECT_PATH, self.exp_name)
#            X = np.genfromtxt(filename,delimiter=",")
#
#            if X.ndim > 1:
#                keep = X[:,1] == itr
#                avg_auc = np.nan_to_num(np.mean(X[:,2][keep]))
#            else:
#                avg_auc = 0.0
#
#            if score >= avg_auc:
#                with open(filename,"a") as f:
#                    f.write("{}, {}, {} \n".format(self.exp_id, itr, score))
#            else:
#                import pdb; pdb.set_trace()
#                endtrain = True
#
#        return endtrain

    @overrides
    def process_samples(self, itr, paths):
        path_lengths = []

        ave_r = []
        ave_kl = []

        ave_col = []
        ave_off = []
        ave_rev = []

        sample_z = []

        if self.model_all:
            new_paths = []
            for path in paths:
                obs, act = path['observations'], path['actions']
                n_cars = obs.shape[-1] / 51 # WARNING: beware magic numbers
                n_cars_ = act.shape[-1] / 2
                assert n_cars == n_cars_
                T = len(path['rewards'])
                o = np.reshape(obs,(-1,T,51))
                a = np.reshape(act,(-1,T,2))
                r = path['env_infos']['rewards']

                mu = np.reshape(path['agent_infos']['mean'],(-1,T,2))
                std = np.reshape(path['agent_infos']['log_std'],(-1,T,2))
                for i in range(o.shape[0]):
                    new_paths.append(
                            dict(
                                observations = o[i],
                                actions = a[i],
                                #rewards = path['rewards'],
                                rewards = r[:,i],
                                agent_infos = {
                                    "mean": mu[i],
                                    "log_std":std[i]
                                    },
                                env_infos = dict(
                                    collision = \
                                            path['env_infos']['collision'][:,i],
                                    offroad = \
                                            path['env_infos']['offroad'][:,i],
                                    reverse = \
                                            path['env_infos']['reverse'][:,i],
                                    domain = path['env_infos']['domain']
                                    )
                                )
                            )
            paths = new_paths

        if self.use_replay_buffer:
            self.replay_buffer.add(paths)
            paths = self.replay_buffer.sample(len(paths))

        for path in paths:
            path['env_rewards'] = path['rewards']
            obs = path['observations']
            act = path['actions']

            infos = path['env_infos']
            path['dom_ix'] = y = infos['dom_ix']


            if self.rew_aug < 1.0:
                r_ga = self.reward_model.compute_reward(obs,act)
                r_rl = path['env_rewards']
                rewards = (r_ga * (1.0-self.rew_aug)) + (r_rl * self.rew_aug)
            else:
                rewards = path['env_rewards']

            if self.info_model is not None:
                sample_z.append(
                        obs[0,-self.info_model.z_dim:].argmax()
                        )
                eos = np.array([obs.shape[0]])
                # forward pass on info model
                _r = self.info_model.compute_reward(obs, act, EOS = eos, Y = y)
                # update rewards and info accuracy
                rewards += _r

            # get collisions
            ave_r.append(rewards.mean())
            try:
                ave_col.append(infos['collision'].mean())
                ave_off.append(infos['offroad'].mean())
                ave_rev.append(infos['reverse'].mean())
            except:
                pass

            path['rewards'] = rewards

            assert not np.isnan(np.sum(rewards))
            path_lengths.append(path['observations'].shape[0])

        assert all([path['rewards'].ndim == 1 for path in paths])
        logger.record_tabular('pathLengths', np.mean(path_lengths))
        logger.record_tabular('ave_reward', np.mean(ave_r))
        logger.record_tabular('ave_kl', np.mean(ave_kl))
        logger.record_tabular('ave_col', np.mean(ave_col))
        logger.record_tabular('ave_off', np.mean(ave_off))
        logger.record_tabular('ave_rev', np.mean(ave_rev))

        self.records['path_lengths'].append(np.mean(path_lengths))

        # sampling prob
        if self.info_model is not None:
            self.record_freqs_and_ent(sample_z, "sample")

        return self.sampler.process_samples(itr, paths)
