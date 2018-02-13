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

from rllab.envs.racing import CarRacing
from rllab.envs.julia_env import JuliaTrackEnv

parser = argparse.ArgumentParser()

# Logger Params
#parser.add_argument('--exp_name',type=str,default='trpo_expert')
parser.add_argument('--tabular_log_file',type=str,default= 'tab.txt')
parser.add_argument('--text_log_file',type=str,default= 'tex.txt')
parser.add_argument('--params_log_file',type=str,default= 'args.txt')
parser.add_argument('--snapshot_mode',type=str,default='all')
parser.add_argument('--log_tabular_only',type=bool,default=False)
parser.add_argument('--log_dir',type=str)
parser.add_argument('--args_data')

parser.add_argument("--n_itr",type=int,default=500)
parser.add_argument("--max_traj_len",type=int,default=200)
parser.add_argument("--batch_size",type=int,default=40 * 200)
parser.add_argument("--environment",type=str,default="Racing-State-Action")

parser.add_argument("--normalize",type=int,default=0)
parser.add_argument("--recurrent",type=int,default=0)

# Network Params
parser.add_argument("--hidden_sizes",type=int,nargs="+",default=[32,32,32,16])
parser.add_argument("--nonlinearity",type=str,default="tanh")

args = parser.parse_args()

#env = TfEnv(normalize(CartpoleEnv())) ## normalize or not ?
nonlin = {"relu":tf.nn.relu,"tanh":tf.nn.tanh,"elu":tf.nn.elu}[args.nonlinearity]
if args.environment == "CartPole":
    env = CartpoleEnv()
elif args.environment == "Pendulum":
    env = gym.make("Pendulum-v0")
elif args.environment == "Racing-State":
    env = CarRacing(mode='state',features=args.features)
elif args.environment == "Racing-State-Action":
    env = CarRacing(mode='state_action',features=args.features)
env = TfEnv(env)

if args.normalize:
    assert False

if args.recurrent:
    feat_net = MLP("feat_net", env.observation_space.shape, args.hidden_sizes[-1], args.hidden_sizes[:-1], nonlin, nonlin)
    policy = GaussianGRUPolicy("policy", env_spec=env.spec, hidden_dim=32,
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
exp_name = args.environment + '-'+str(c)

while exp_name in os.listdir(model_path+'/'+date+'/'):
    c += 1
    exp_name = args.environment + '-'+str(c)

exp_dir = date+'/'+exp_name
log_dir = osp.join(config.LOG_DIR, exp_dir)

policy.set_log_dir(log_dir)

runner = RLLabRunner(algo, args, exp_dir)
#policy.save_extra_data(["initial_obs_mean","initial_obs_std"],[initial_obs_mean, initial_obs_std])
runner.train()

#run_experiment_lite(
    #algo.train(),
    #n_parallel=4,
    #seed=1,
#)
