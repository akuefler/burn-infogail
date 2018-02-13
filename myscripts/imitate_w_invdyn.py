import argparse
import calendar
import os
import os.path as osp
from rllab import config

import gym
import tensorflow as tf
import numpy as np

from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.box2d.cartpole_env import CartpoleEnv
from rllab.envs.normalized_env import normalize
from tf_rllab.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer
from tf_rllab.optimizers.conjugate_gradient_optimizer import FiniteDifferenceHvp
from tf_rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy

from tf_rllab import RLLabRunner

from rllab.config_personal import expert_trajs_path, model_path

from rllab.envs.gym_env import GymEnv

from tf_rllab.baselines.gaussian_conv_baseline import GaussianConvBaseline

from trn.algo import GAIL

import rltools.util

from myscripts import _create_env, _create_policy, _create_expert_data, _create_aux_networks, _create_log

import h5py

parser = argparse.ArgumentParser()

# Logger Params
parser.add_argument('--exp_name',type=str,default='GAIL')
parser.add_argument('--tabular_log_file',type=str,default= 'tab.txt')
parser.add_argument('--text_log_file',type=str,default= 'tex.txt')
parser.add_argument('--params_log_file',type=str,default= 'args.txt')
parser.add_argument('--snapshot_mode',type=str,default='none')
parser.add_argument('--log_tabular_only',type=bool,default=False)
parser.add_argument('--log_dir',type=str)
parser.add_argument('--args_data')

parser.add_argument('--save_models',type=str,nargs="+",default=["policy"])
parser.add_argument('--itr_threshold',type=int,default=100)
parser.add_argument('--reward_threshold',type=float,default=-1000.)

# reinforcement learning params
parser.add_argument("--n_itr",type=int,default=250)
parser.add_argument("--discount",type=float,default=0.99)
parser.add_argument("--baseline_type",type=str,default="linear")

# architecture params
parser.add_argument("--policy_nonlinearity",type=str,default="tanh")
parser.add_argument("--info_nonlinearity",type=str,default="tanh")
parser.add_argument("--reward_nonlinearity",type=str,default="tanh")
parser.add_argument("--adaptive_std",type=int,default=0)

# policy params
parser.add_argument("--policy_recur_dim",type=int,default=32)
parser.add_argument("--policy_cell",type=str,default="none")
parser.add_argument("--policy_hidden_sizes",type=int,nargs="+",default=[128,128])
parser.add_argument("--bc_init",type=int,default=0)
parser.add_argument("--prior_type",type=str,default="discrete")
parser.add_argument("--end_on_failure",type=int,default=1)

# network params
parser.add_argument("--reward_trainer",type=str,default="adam")
parser.add_argument("--info_trainer",type=str,default="adam")
parser.add_argument("--reward_learning_rate",type=float,default=0.0001)
parser.add_argument("--info_learning_rate",type=float,default=0.0001)

parser.add_argument("--info_drop_prob",type=float,default=0.0)
parser.add_argument("--reward_drop_prob",type=float,default=0.0)

parser.add_argument("--info_stochastic",type=int,default=0)

# trpo params
parser.add_argument("--trpo_step_size",type=float,default=0.01)
parser.add_argument("--trpo_batch_size",type=int,default=50 * 100)

# reward params
parser.add_argument("--rew_aug",type=float,default=0.0)
parser.add_argument("--reward_hidden_sizes",type=int,nargs="+",default=[32])
parser.add_argument("--reward_epoch",type=int,default=1)
parser.add_argument("--clip",type=float,default=0.01)
parser.add_argument("--wgan",type=int,default=0)
parser.add_argument("--gail_batch_size",type=int,default=150)
parser.add_argument("--reward_batch_size",type=int,default=10)

# inverse dynamics model
parser.add_argument("--invdyn_hidden_sizes",type=int,nargs='+',default=[32,32,32])
parser.add_argument("--invdyn_start_epoch",type=int,default=0)
parser.add_argument("--invdyn_epoch",type=int,default=50)
parser.add_argument("--invdyn_temperature",type=float,default=1.0)
parser.add_argument("--environment",type=str,default='JTZM')

# vae params
parser.add_argument("--z_dim",type=int,default=2)
parser.add_argument("--kl_weight",type=float,default=0.001)

parser.add_argument("--policy_merge",type=str,default="mul")
parser.add_argument("--z_policy_merge_idx",type=int,default=1)
parser.add_argument("--z_policy_hidden_sizes",type=int,nargs="+",default=[])

parser.add_argument("--z_reward_merge_idx",type=int,default=1)
parser.add_argument("--z_reward_hidden_sizes",type=int,nargs="+",default=[])

# info params
parser.add_argument("--use_infogail",type=int,default=1)
parser.add_argument("--use_info_prior", type=int, default=0)

parser.add_argument("--info_reg",type=float,default=0.1)
parser.add_argument("--info_ent",type=float,default=0.0)
parser.add_argument("--info_cnf",type=float,default=0.0)

parser.add_argument("--info_decay_rate",type=float,default=0.99)
parser.add_argument("--info_decay_step",type=int,default=5)

parser.add_argument("--z_discrete",type=int,default=1)

parser.add_argument("--info_epoch",type=int,default=5)
parser.add_argument("--info_hidden_sizes",type=int,nargs="+",default=[128,128])
parser.add_argument("--info_recur_dim",type=int,default=0)
parser.add_argument("--info_cell",type=str,default="gru")
parser.add_argument("--info_batch_size",type=int,default=10)

parser.add_argument("--cnf_hidden_sizes",type=int,nargs="+",default=[])

parser.add_argument("--use_replay_buffer",type=int,default=0)
parser.add_argument("--include_cnf_in_reward",type=int,default=0)

# curriculum params
parser.add_argument("--curr_start",type=int,default=1)
parser.add_argument("--curr_add",type=int,default=0)
parser.add_argument("--curr_step",type=int,default=1)

parser.add_argument("--use_valid",type=int,default=0)

###################
# JULIA NGSIM PARAMS
###################
parser.add_argument("--max_path_length",type=int,default=50)
parser.add_argument("--trajdata_indices",type=int,nargs="+",default=[1, 2, 3, 4, 5, 6])
parser.add_argument("--domain_indices",type=int,nargs="+",default=[0, 1, 2, 3])

###################
# JULIA TRACK PARAMS
###################
parser.add_argument("--mix_data_classes",type=int,default=0)
parser.add_argument("--model_all",type=int,default=0)
parser.add_argument("--index_features",type=int,default=0)

###############
# RACING PARAMS
###############
parser.add_argument("--track_turn_rate",type=float,default=0.31)
parser.add_argument("--track_width",type=int,default=40)
parser.add_argument("--friction_limit",type=int,default=1000000)
parser.add_argument("--wheel_moment_of_inertia",type=int,default=4000)
parser.add_argument("--engine_power",type=int,default=100000000)
parser.add_argument("--brake_force",type=int,default=15)
parser.add_argument("--road_friction",type=float,default=0.9)
parser.add_argument("--grass_friction",type=float,default=0.6)
# racing features
parser.add_argument("--features",type=str,nargs="+",default=["vel", "curves", "on_grass", "lane_offset", "heading_angle"])

#################
# PENDULUM PARAMS
#################
parser.add_argument("--max_speed",type=float,default=10.)
parser.add_argument("--max_torque",type=float,default=4.)
parser.add_argument("--g",type=float,default=10.)
parser.add_argument("--m",type=float,default=1.)
parser.add_argument("--l",type=float,default=1.)

###############
# LANDER PARAMS
###############
parser.add_argument("--main_engine_power",type=float,default=13.0)
parser.add_argument("--side_engine_power",type=float,default=0.6)
parser.add_argument("--side_engine_height",type=float,default=14.0)
parser.add_argument("--side_engine_away",type=float,default=12.0)
parser.add_argument("--lander_density",type=float,default=5.0)

# misc params
parser.add_argument("--debug",type=int,default=0)
parser.add_argument("--seed",type=int,default=456)
parser.add_argument("--normalize",type=int,default=0)
#parser.add_argument("--expert_data_path",type=str,default="expert_trajs/racing/Racing-State-0")
#parser.add_argument("--expert_data_path",type=str,default="models/17-03-10/PP-EXPERT-G10-0")

args = parser.parse_args()
args.reward_threshold = int(args.reward_threshold)

if not args.use_infogail:
    args.z_dim = 0

assert (not args.wgan and args.reward_epoch == 1) or \
       (args.wgan and args.reward_epoch > 1)

def main():
    np.random.seed(args.seed)
    tf.set_random_seed(args.seed)

    # create the environment
    env = _create_env(args)

    # create expert data
    expert_data_T, expert_data_V = _create_expert_data(args)
    expert_data = dict(
            train = expert_data_T,
            valid = expert_data_V
            )

    # create policy
    policy, init_ops = _create_policy(args, env)

    # create auxiliary networks (invdyn, reward, variational posterior)
    invdyn_model, reward_model, info_model, env = _create_aux_networks(args, env)

    # create baseline
    if args.baseline_type == "linear":
        baseline = LinearFeatureBaseline(env_spec=None)
    else:
        assert False

    # use date and time to create new logging directory for each run
    date= calendar.datetime.date.today().strftime('%y-%m-%d')
    if date not in os.listdir(model_path):
        os.mkdir(model_path+'/'+date)

    c = 0
    exp_name = '{}-'.format(args.exp_name) + str(c)

    while exp_name in os.listdir(model_path+'/'+date+'/'):
        c += 1
        exp_name = '{}-'.format(args.exp_name)+str(c)

    exp_dir = date+'/'+exp_name
    log_dir = osp.join(config.LOG_DIR, exp_dir)

    policy.set_log_dir(log_dir)
    if info_model is not None:
        info_model.set_log_dir(log_dir)

    _create_log(args)

    # run GAIL algorithm
    models = {"policy":policy, "info":info_model, "reward":reward_model}
    bpo_args = dict(
        n_itr=args.n_itr,
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=args.trpo_batch_size,
        max_path_length=args.max_path_length,
        discount=args.discount,
        step_size=args.trpo_step_size,
        force_batch_sampler=True,
        whole_paths=True,
        init_ops=init_ops,
        optimizer=ConjugateGradientOptimizer(hvp_approach=FiniteDifferenceHvp(base_eps=1e-5)),
        save_models=[models[model_name] for model_name in args.save_models]
        )
    vae_args = dict(
            kl_weight=args.kl_weight,
            )
    curriculum = dict(
            start = args.curr_start,
            add = args.curr_add,
            step = args.curr_step
            )
    if not args.model_all : curriculum = {}
    kwargs = {k:v for k, v in bpo_args.items() + vae_args.items()}
    algo = GAIL(
                args.exp_name,
                exp_name,
                expert_data,
                reward_model,
                args.gail_batch_size,
                invdyn_model=invdyn_model,
                info_model=info_model,
                debug=args.debug,
                model_all=args.model_all,
                curriculum=curriculum,
                rew_aug=args.rew_aug,
                use_replay_buffer=args.use_replay_buffer,
                **kwargs
                )

    runner = RLLabRunner(algo, args, exp_dir)
    runner.train()

if __name__ == "__main__":
    main()
