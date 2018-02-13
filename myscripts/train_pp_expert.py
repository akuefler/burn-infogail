import argparse
import calendar
import os
import os.path as osp
from rllab import config

import gym
import tensorflow as tf

from tf_rllab.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.box2d.cartpole_env import CartpoleEnv
from rllab.envs.normalized_env import normalize
from tf_rllab.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer
from tf_rllab.optimizers.conjugate_gradient_optimizer import FiniteDifferenceHvp

from tf_rllab.core.network import MLP
from tf_rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from tf_rllab.policies.gaussian_gru_policy import GaussianGRUPolicy

from rllab.envs.tf_env import TfEnv

from tf_rllab import RLLabRunner

from rllab.config_personal import expert_trajs_path, model_path

from rllab.envs.parameterized_ll import LunarLander, LunarLanderContinuous
from rllab.envs.parameterized_pendulum import ParameterizedPendulumEnv

parser = argparse.ArgumentParser()

# Logger Params
parser.add_argument('--exp_name',type=str,default='')
parser.add_argument('--tabular_log_file',type=str,default= 'tab.txt')
parser.add_argument('--text_log_file',type=str,default= 'tex.txt')
parser.add_argument('--params_log_file',type=str,default= 'args.txt')
parser.add_argument('--snapshot_mode',type=str,default='all')
parser.add_argument('--log_tabular_only',type=bool,default=False)
parser.add_argument('--log_dir',type=str)
parser.add_argument('--args_data')

parser.add_argument("--n_itr",type=int,default=500)
parser.add_argument("--max_traj_len",type=int,default=300)
parser.add_argument("--batch_size",type=int,default=40 * 200)
parser.add_argument("reward_batch_size",type=int,default=10)

parser.add_argument("--normalize",type=int,default=0)
parser.add_argument("--recurrent",type=int,default=0)

# Network Params
parser.add_argument("--hidden_sizes",type=int,nargs="+",default=[32,32])
parser.add_argument("--nonlinearity",type=str,default="tanh")

# Simulator params
parser.add_argument("--environment",type=str,default="LL")

## Pendulum
parser.add_argument("--max_speed",type=float,default=10.)
parser.add_argument("--max_torque",type=float,default=4.)
parser.add_argument("--g",type=float,default=10.)
parser.add_argument("--m",type=float,default=1.)
parser.add_argument("--l",type=float,default=1.)

## Lunar Lander
parser.add_argument("--main_engine_power",type=float,default=13.0)
parser.add_argument("--side_engine_power",type=float,default=0.6)
parser.add_argument("--side_engine_height",type=float,default=14.0)
parser.add_argument("--side_engine_away",type=float,default=12.0)
parser.add_argument("--lander_density",type=float,default=5.0)

# Racing Features
parser.add_argument("--features",type=str,nargs="+",default=["pos","vel","abs","hull_ang","wheel_ang",
                                                             "reward","hull_ang_vel","speed","xy_dist_from_road"])

args = parser.parse_args()

nonlin = {"relu":tf.nn.relu,"tanh":tf.nn.tanh,"elu":tf.nn.elu}[args.nonlinearity]
if args.environment == "PP":
    env = ParameterizedPendulumEnv(max_speed=args.max_speed, max_torque=args.max_torque,
                                   g=args.g, m=args.m, l=args.l)
elif args.environment == "LL":
    env = LunarLanderContinuous(main_engine_power=args.main_engine_power,
                                side_engine_power=args.side_engine_power,
                                side_engine_height=args.side_engine_height,
                                side_engine_away=args.side_engine_away,
                                lander_density=args.lander_density)
env = TfEnv(env)

if args.normalize:
    env = TfEnv(normalize(env, normalize_obs= True, running_obs=True))

if args.recurrent:
    feat_net = MLP("feat_net", env.observation_space.shape, args.hidden_sizes[-2], args.hidden_sizes[:-2], nonlin, nonlin)
    policy = GaussianGRUPolicy("policy", env_spec=env.spec, hidden_dim=hidden_sizes[-1],
                               feature_network=feat_net,
                               state_include_action=False,
                               hidden_nonlinearity=nonlin)
else:
    policy = GaussianMLPPolicy(
        name="policy",
        env_spec=env.spec,
        # The neural network policy should have two hidden layers, each with 32 hidden units.
        hidden_sizes=args.hidden_sizes,
        hidden_nonlinearity=nonlin
    )

baseline = LinearFeatureBaseline(env_spec=env.spec)

algo = TRPO(
    env=env,
    policy=policy,
    baseline=baseline,
    batch_size=args.batch_size,
    max_path_length=args.max_traj_len,
    n_itr=args.n_itr,
    discount=0.99,
    step_size=0.01,
    #force_batch_sampler=True,
    optimizer=ConjugateGradientOptimizer(hvp_approach=FiniteDifferenceHvp(base_eps=1e-5))
)

# use date and time to create new logging directory for each run
date= calendar.datetime.date.today().strftime('%y-%m-%d')
if date not in os.listdir(model_path):
    os.mkdir(model_path+'/'+date)

c = 0
exp_name = 'ParamEnv-{}-'.format(args.environment) + str(c)
while exp_name in os.listdir(model_path+'/'+date+'/'):
    c += 1
    exp_name = 'ParamEnv-{}-'.format(args.environment) + str(c)

exp_dir = date+'/'+exp_name
log_dir = osp.join(config.LOG_DIR, exp_dir)

policy.set_log_dir(log_dir)

runner = RLLabRunner(algo, args, exp_dir)
#policy.save_extra_data(["initial_obs_mean","initial_obs_std"],[initial_obs_mean, initial_obs_std])
runner.train()
