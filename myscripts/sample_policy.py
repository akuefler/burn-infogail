from __future__ import absolute_import, print_function

import argparse
import json

import tensorflow as tf

import gym
from rllab.sampler.utils import rollout, gif_rollout, rollout_w_truth #, multi_rollout
from rllab.envs.tf_env import TfEnv

from tf_rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy

from rllab.envs.gym_env import GymEnv
from rllab.envs.racing import CarRacing
from rllab.envs.parameterized_pendulum import ParameterizedPendulumEnv
from rllab.envs.parameterized_ll import LunarLanderContinuous
from rllab.envs.normalized_env import normalize

import numpy as np
import h5py
import os

from rllab import config

from trn.config import best_epochs

import matplotlib.pyplot as plt

from myscripts import _create_env, _create_policy, _restore_model_args, \
nan_stack_ragged_array, gather_dicts, _create_aux_networks, \
_restore_baseline_args, _create_encoder

#MEASURES = ["measure/speed", "measure/laneOffsetL", "measure/laneOffsetR", "measure/POSFT"]
MEASURES = ["measure/speed", "measure/pos_x", "measure/pos_y", "measure/pos", "measure/posFt"]
def compute_error(trajbatch, truth, headers=MEASURES, mpl=300):
    E = []
    M = []
    TR = []
    for key in MEASURES:
        if key == "measure/pos":
            e = (trajbatch['env_infos']['state']['measure/pos_x'] -
                    np.array(truth['measure/pos_x'][:mpl])) ** 2.
            e += (trajbatch['env_infos']['state']['measure/pos_y'] -
                    np.array(truth['measure/pos_y'][:mpl])) ** 2.
        else:
            e = (trajbatch['env_infos']['state'][key] -
                    np.array(truth[key][:mpl])) ** 2.
            M.append(trajbatch['env_infos']['state'][key])
            TR.append(truth[key])
        E.append(e)
    E = np.row_stack(E)
    M = np.row_stack(M)
    TR = np.row_stack(TR)
    return E, M, TR

def main():
    parser = argparse.ArgumentParser()

    #parser.add_argument('--epoch', type=int, default= 96)  # epoch file
    parser.add_argument('--end_on_failure',type=int,default=0)
    parser.add_argument('--exp_name', type=str, default=
            #"models/17-06-13/CORL2-06130728-JTZM-8")
            #"models/17-06-13/CORL2-06130728-JTZM-4")
            "models/17-06-13/CORL2-06130728-JTZM-7")
            #"models/17-06-17/CORL3-06172024-JTZM-0")
            #"models/17-06-19/CORL4-06182344-JTZM-1")
    parser.add_argument("--batch_size",type=int,default=1000)
    parser.add_argument("--render",type=int,default=0)
    parser.add_argument("--save",type=int,default=1)
    parser.add_argument("--save_gifs",type=int,default=0)

    parser.add_argument("--max_path_length",type=int,default=300)
    parser.add_argument("--deterministic", type=int, default=0)
    parser.add_argument("--use_info_prior", type=int, default=0)
    parser.add_argument("--denorm", type=int, default=0)

    parser.add_argument("--use_valid", type=int, default=1)

    parser.add_argument("--scene_seed", type=int, default=-1)

    args = parser.parse_args()

    plt.ion()
    info_model = None
    if args.exp_name not in ["vae","bc","random"]:
        exp_name = '../data/{}'.format(args.exp_name)
        args = _restore_model_args(exp_name, args, exclude_keys =
                ["end_on_failure","use_info_prior","use_valid",
                    "max_path_length"])
    else:
        exp_name = '../data/baselines/{}'.format(args.exp_name)
        args = _restore_baseline_args(args)
        if args.use_info_prior:
            info_model = _create_encoder()

    env = _create_env(args, encoder = info_model)
    policy, init_ops = _create_policy(args,env)

    if args.exp_name not in ["vae","bc","random"]:
        _, reward_model, info_model, env = _create_aux_networks(args, env)

    B = args.batch_size
    Do = np.prod(env.observation_space.shape)
    Da = env.action_dim

    exobs = []
    exa = []
    exlen = []
    zs = []
    z_means = []
    z_stds = []
    states = []

    with tf.Session() as sess:
        if args.exp_name == "random":
            sess.run(tf.initialize_all_variables())
        else:
            policy.load_params(exp_name, best_epochs.get(exp_name,-1), [])

        if info_model is not None:
            info_model.load_params(exp_name, best_epochs.get(exp_name,-1), [])

        if args.deterministic:
            policy.deterministic = True

        ERRORS = []
        MEASURE_FEATURES = []
        TRUE_FEATURES = []
        CLASSES = []
        ZS = []

        OFFROADS = []
        COLLISIONS = []
        REVERSALS = []
        LENGTHS = []
        for i in xrange(args.batch_size):
            if i % 100 == 0:
                print("{} of {}".format(i,args.batch_size))
            filename = None
            if args.save_gifs:
                print("Using gif rollout ...")
                gif_directory = "{}/gifs/".format(exp_name)
                if not os.path.isdir(gif_directory):
                    os.mkdir(gif_directory)
                filename = gif_directory + \
                    "gif_valid{}_determin{}_{}".format(args.use_valid,
                            args.deterministic,i)

            #trajbatch = rollout(env, policy, T, animated= args.render)
            if args.exp_name in ["bc","vae"] and args.denorm:
                a_mean = np.array([-0.54467528, 0.13369287])
                a_std = np.array([1.61894407, 0.19208458])
            else:
                a_mean = np.zeros(2)
                a_std = np.ones(2)
            if args.scene_seed < 0:
                seed = i
            else:
                seed = args.scene_seed
            trajbatch, truth = rollout_w_truth(env, policy,
                    args.max_path_length,
                    animated= args.render,
                    mean=a_mean,
                    std=a_std,
                    save_gif = args.save_gifs,
                    filename = filename,
                    seed = seed
                    )

            OFFROADS.append(trajbatch['env_infos']['offroad'].mean())
            COLLISIONS.append(trajbatch['env_infos']['collision'].mean())
            REVERSALS.append(trajbatch['env_infos']['reverse'].mean())
            LENGTHS.append(len(trajbatch['env_infos']['collision']))

            E, M, TR = compute_error(trajbatch, truth, mpl=args.max_path_length)
            ERRORS.append(E[None,...])
            MEASURE_FEATURES.append(M)
            TRUE_FEATURES.append(TR)
            try:
                ZS.append(trajbatch["env_infos"]["z"].argmax())
            except:
                pass

            exobs.append( trajbatch['observations'] )
            exa.append( trajbatch['actions'] )
            exlen.append( len(trajbatch['observations']) )
            try:
                z_means.append( trajbatch["env_infos"]["z_mean"][0] )
                z_stds.append( trajbatch["env_infos"]["z_std"][0] )
            except KeyError:
                pass

            states.append( trajbatch["env_infos"]["state"])

    print("Avg. Offroad")
    print(np.mean(OFFROADS))

    print("Avg. Coll")
    print(np.mean(COLLISIONS))

    print("Avg. Rev.")
    print(np.mean(REVERSALS))

    ER = np.sqrt( np.mean(np.concatenate(ERRORS,axis=0),axis=0) )

    exobs_B_T_Do = nan_stack_ragged_array(exobs,args.max_path_length)
    exa_B_T_Da = nan_stack_ragged_array(exa,args.max_path_length)
    exlen_B = np.array(exlen)
    states_D = gather_dicts(states, args.max_path_length)
    if args.z_dim > 0:
        zmean_B_Dz = np.row_stack(z_means)
        zstd_B_Dz = np.row_stack(z_stds)

    if args.save:
        if not os.path.isdir("{}/train".format(exp_name)):
            os.mkdir("{}/train".format(exp_name))
        if not os.path.isdir("{}/valid".format(exp_name)):
            os.mkdir("{}/valid".format(exp_name))

        expert_traj_path = \
            "{}/{}/expert_trajs.h5".format(exp_name,["train","valid"][args.use_valid])
        expert_errors_path = \
            "{}/{}/expert_errors_denorm{}_infoprior{}_deterministic{}.h5".format(exp_name,["train","valid"][args.use_valid],
                    args.denorm, args.use_info_prior, args.deterministic)

        if os.path.isfile(expert_traj_path):
            os.remove(expert_traj_path)
        if os.path.isfile(expert_errors_path):
            os.remove(expert_errors_path)

        with h5py.File(expert_errors_path,"a") as hf:
            hf.create_dataset("measures",data=MEASURES)
            hf.create_dataset("measure_features",data=MEASURE_FEATURES)
            hf.create_dataset("true_features",data=TRUE_FEATURES)
            hf.create_dataset("classes",data=CLASSES)
            hf.create_dataset("errors",data=ERRORS)
            hf.create_dataset("RMSE",data=ER)
            hf.create_dataset("z",data=ZS)

            hf.create_dataset("offroads",data=OFFROADS)
            hf.create_dataset("collisions",data=COLLISIONS)
            hf.create_dataset("reversals",data=REVERSALS)

        with h5py.File(expert_traj_path,"a") as hf:
            hf.create_dataset("a_B_T_Da",data=exa_B_T_Da)
            hf.create_dataset("obs_B_T_Do",data=exobs_B_T_Do)
            hf.create_dataset("len_B",data=exlen_B)

            if args.z_dim > 0:
                hf.create_dataset("zmean_B_T_Dz",data=zmean_B_Dz)
                hf.create_dataset("zstd_B_T_Dz",data=zstd_B_Dz)

            for key, val in states_D.items():
                hf.create_dataset("state/{}_B_T_Ds".format(key), data=val)

if __name__ == '__main__':
    main()

