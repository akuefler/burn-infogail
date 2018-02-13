# Example: ./simple_trpo_example.py --env CartPole-v0 --n_iter 60 --save_freq 30 --log cart_pole.h5

from __future__ import absolute_import, print_function

import argparse
import json

import tensorflow as tf

import gym
from rllab.sampler.utils import rollout
from rllab.envs.tf_env import TfEnv

from tf_rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline

from rllab.envs.gym_env import GymEnv
from rllab.envs.racing import CarRacing
from rllab.envs.parameterized_pendulum import ParameterizedPendulumEnv
from rllab.envs.parameterized_ll import LunarLanderContinuous
from rllab.envs.normalized_env import normalize

import numpy as np
import h5py
import os

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--epoch', type=int, default= -1)  # epoch file
    parser.add_argument('--exp_name', type=str, default= "models/17-04-10/RS-0")  # log file
    parser.add_argument("--batch_size",type=int,default=1500)

    parser.add_argument("--render",type=int,default=0)
    parser.add_argument("--save",type=int,default=1)

    parser.add_argument("--max_traj_len",type=int,default=None)

    args = parser.parse_args()

    exp_name = '../data/{}'.format(args.exp_name)
    with open('{}/args.txt'.format(exp_name),'r') as f:
        model_args = ''.join(f.readlines())
        model_args = model_args.replace("null","None")
        model_args = model_args.replace("false","False")
        model_args = model_args.replace("true","True")
        model_args = eval(model_args)

    nonlin = {"tanh":tf.nn.tanh,"relu":tf.nn.relu,"elu":tf.nn.elu}[model_args['nonlinearity']]
    environment = model_args['environment']
    if environment == "CartPole":
        env = TfEnv(CartpoleEnv())
    elif environment == "Pendulum":
        env = gym.make("Pendulum-v0")
        env = TfEnv(env)
    elif environment == "NoisyPendulum":
        gym.envs.register(
            id="NoisyPendulum-v0",
            entry_point='rllab.envs.target_env:NoisyPendulum',
            timestep_limit=999,
            reward_threshold=195.0,
        )
        env = TfEnv(GymEnv("NoisyPendulum-v0"))
    elif environment == "RS":
        env = CarRacing(mode='state',**model_args)
        env = TfEnv(env)
    elif environment == "RA":
        env = CarRacing(mode='state_state',**model_args)
        env = TfEnv(env)
    elif environment == "PP":
        env = ParameterizedPendulumEnv(**model_args)
        env = TfEnv(env)
    elif environment == "LL":
        env = LunarLanderContinuous(**model_args)
        env = TfEnv(env)

    #if model_args['normalize']:
    if False:
        filename = "{}/epochs.h5".format(exp_name)
        with h5py.File(filename,"r") as hf:
            keys = hf.keys()
            obs_mean = hf[keys[args.epoch]]['obs_mean'][...]
            obs_var = hf[keys[args.epoch]]['obs_var'][...]
        env = TfEnv(normalize(env, normalize_obs=True, initial_obs_mean=obs_mean, initial_obs_var=obs_var))

    try:
        policy = GaussianMLPPolicy(
            name="policy",
            env_spec=env.spec,
            hidden_nonlinearity=nonlin,
            # The neural network policy should have two hidden layers, each with 32 hidden units.
            hidden_sizes=model_args['hidden_sizes'])
    except:
        policy = GaussianMLPPolicy(
            name="policy",
            env_spec=env.spec,
            #hidden_nonlinearity=nonlin,
            hidden_sizes=model_args['policy_hidden_sizes'])

    baseline = LinearFeatureBaseline(env_spec=env.spec)

    B = args.batch_size
    Do = np.prod(env.observation_space.shape)
    Da = env.action_dim
    if args.max_traj_len is None:
        T = model_args['max_traj_len']
    else:
        T = 100

    exa_B_T_Da = np.zeros((B,T,Da))
    exobs_B_T_Do = np.zeros((B,T,Do))
    exlen_B = np.zeros(B)

    with tf.Session() as sess:
        #sess.run(tf.initialize_all_variables())
        policy.load_params(exp_name, args.epoch, [])

        for i in xrange(args.batch_size):
            print("{} of {}".format(i,args.batch_size))
            #env, policy, max_traj_len, action_space
            trajbatch= rollout(env, policy, T, animated= args.render)

            exobs_B_T_Do[i] = trajbatch['observations']
            exa_B_T_Da[i] = trajbatch['actions']
            exlen_B[i] = T

    if args.save:
        expert_traj_path = "{}/expert_trajs.h5".format(exp_name)
        if os.path.isfile(expert_traj_path):
            os.remove(expert_traj_path)
        with h5py.File(expert_traj_path,"a") as hf:
            hf.create_dataset("a_B_T_Da",data=exa_B_T_Da)
            hf.create_dataset("obs_B_T_Do",data=exobs_B_T_Do)
            hf.create_dataset("len_B",data=exlen_B)
            if model_args['normalize']:
                hf.create_dataset("obs_mean",data=obs_mean)
                hf.create_dataset("obs_var",data=obs_var)

if __name__ == '__main__':
    main()
