import tensorflow as tf
import numpy as np

from trn.policy import MergeMLP, FactoredMLPPolicy
from tf_rllab.core.network import MLP
from tf_rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from tf_rllab.policies.gaussian_gru_policy import GaussianGRUPolicy
from tf_rllab.policies.gaussian_lstm_policy import GaussianLSTMPolicy

from rllab.envs.tf_env import TfEnv
from rllab.config import PROJECT_PATH

#from rllab.envs.racing import CarRacing
#from rllab.envs.parameterized_pendulum import ParameterizedPendulumEnv
#from rllab.envs.parameterized_ll import LunarLanderContinuous
from rllab.envs.julia_env import JuliaTrackEnv, JuliaNGSIMEnv

from trn.config import expert_data_paths, policy_paths
from trn.network import RewardModel, FactoredRewardModel,\
    InverseDynamicsModel,RE_InfoModel, FF_InfoModel, Encoder

import itertools

import rltools.util

import h5py

import collections

NONLIN = {"tanh":tf.nn.tanh,"relu":tf.nn.relu,"elu":tf.nn.elu}
TRAINER = {"adam":tf.train.AdamOptimizer,
           "sgd":tf.train.GradientDescentOptimizer,
           "rmsprop":tf.train.RMSPropOptimizer}
CELL = {"gru":tf.nn.rnn_cell.GRUCell, "lstm":tf.nn.rnn_cell.LSTMCell, "none":None}
PI_CELL = {"gru":GaussianGRUPolicy, "lstm":GaussianLSTMPolicy, "none":GaussianMLPPolicy}

def nan_stack_ragged_array(X, mpl):
    Y = []
    for x in X:
        if x.ndim == 1:
            x = x[...,None]
        t, d = x.shape
        if t < mpl:
            try:
                x = np.concatenate([x, np.ones((mpl-t,d)) * np.nan], axis=0)
            except TypeError:
                import pdb; pdb.set_trace()
        Y.append(x[None,...])
    return np.row_stack(Y)

def gather_dicts(dicts, mpl):
    """
    turns list of dicts into dict of lists
    """
    result = collections.defaultdict(list)
    for d in dicts:
        for k, v in d.items():
            result[k].append(v)

    result = {k: np.array(v) for k, v in result.items()}
    result = {k: nan_stack_ragged_array(v, mpl) for k, v in result.items()}

    return result

def _restore_model_args(exp_name, args, exclude_keys = []):
    with open('{}/args.txt'.format(exp_name),'r') as f:
        model_args = ''.join(f.readlines())
        model_args = model_args.replace("null","None")
        model_args = model_args.replace("false","False")
        model_args = model_args.replace("true","True")
        model_args = eval(model_args)

    for key in exclude_keys:
        if key in model_args:
            del model_args[key]

    # update argument dictionary from model.
    args.__dict__.update(model_args)
    return args

def _create_encoder():
    info_model = Encoder()
    return info_model

def _restore_baseline_args(args):
#    with open('{}/args.txt'.format(exp_name),'r') as f:
#        model_args = ''.join(f.readlines())
#        model_args = model_args.replace("null","None")
#        model_args = model_args.replace("false","False")
#        model_args = model_args.replace("true","True")
#        model_args = eval(model_args)
#
#    for key in exclude_keys:
#        if key in model_args:
#            del model_args[key]
    model_args = dict(
            use_infogail=1,
            domain_indices = [1],
            mix_data_classes = 1,
            model_all = 0,
            environment = "JTZM",
            prior_type = "standard",
            curr_start = 1,
            index_features = False,
            adaptive_std = False,
            policy_recur_dim = 0,
            bc_init = False,
            reward_trainer="adam",
            reward_learning_rate=0.0,
            reward_hidden_sizes=[],
            reward_batch_size=0,
            reward_nonlinearity="tanh",
            reward_epoch=0,
            wgan=0,
            clip=0.0,
            info_trainer="adam",
            info_nonlinearity="tanh",
            info_cell="none",
            info_recur_dim=0,
            info_learning_rate=0.0,
            info_hidden_sizes=[],
            info_batch_size=1,
            info_epoch=0,
            info_reg=0.0,
            info_cnf=0.0,
            info_ent=0.0,
            invdyn_start_epoch=0,
            # relevant settings ... 
            z_dim = 2,
            policy_hidden_sizes = [128, 128],
            policy_merge = "concat",
            policy_cell = "none",
            policy_nonlinearity = "relu",
            z_policy_merge_idx = 0,
            z_policy_hidden_sizes = []
            )

    # update argument dictionary from model.
    args.__dict__.update(model_args)
    return args

def _create_expert_data(args, path = None):
    domain_indices = None
    if path is None:
        if "JTZ" in args.environment:
            if args.mix_data_classes:
                path = expert_data_paths["JTZM"]
            else:
                path = expert_data_paths["JTZS"]

            assert(args.max_path_length == 50)
            domain_indices = args.domain_indices
        else:
            path = expert_data_paths[args.environment]

    print("Loading trajs from: {}".format(path))
    expert_data_T, _ = \
        rltools.util.load_trajs("../data/{}/train/{}expert_trajs.h5".format(path,["","multiagent_"][args.model_all]),
                12000, domain_indices=domain_indices, swap=False)
    expert_data_V, _ = \
        rltools.util.load_trajs("../data/{}/valid/{}expert_trajs.h5".format(path,["","multiagent_"][args.model_all]),
                12000, domain_indices=domain_indices, swap=False)

    return expert_data_T, expert_data_V

def _create_log(args):
    fo = open("{}/logs/{}.txt".format(PROJECT_PATH,args.exp_name),"a")
    fo.close()

def _create_env(args, encoder = None):
    """
    select and wrap the learning environment.
    """
    if args.environment in ["RS", "RA"]:
        if args.environment == "RS":
            mode = 'state'
        elif args.environment == 'RA':
            mode = 'state_action'
        env = CarRacing(mode=mode,features=args.features,
                        track_turn_rate=args.track_turn_rate,
                        track_width=args.track_width,
                        friction_limit=args.friction_limit,
                        wheel_moment_of_inertia=args.wheel_moment_of_inertia,
                        engine_power=args.engine_power,
                        brake_force=args.brake_force,
                        road_friction=args.road_friction, grass_friction=args.grass_friction
                        )
    elif args.environment == 'PP':
        env = ParameterizedPendulumEnv(max_speed=args.max_speed, max_torque=args.max_torque,
                                       g=args.g, m=args.m, l=args.l)
    elif args.environment == 'LL':
        env = LunarLanderContinuous(main_engine_power=args.main_engine_power,
                                   side_engine_power=args.side_engine_power,
                                   side_engine_height=args.side_engine_height,
                                   side_engine_away=args.side_engine_away,
                                   lander_density=args.lander_density)

    elif args.environment in ["JTZS", "JTZM"]:
        mix_class = (args.environment == "JTZM")
        env_dict = dict(
                domain_indices = args.domain_indices,
                mix_class = bool(args.mix_data_classes),
                model_all = bool(args.model_all),
                use_valid = bool(args.use_valid)
                )
        env = JuliaTrackEnv(env_dict, z_dim=args.z_dim, normalize_obs = True,
                prior_type = args.prior_type,
                n_egos = args.curr_start,
                index_features = args.index_features,
                end_on_failure = bool(args.end_on_failure))

    elif args.environment == "JNGSIM":
        env_dict = dict(
                trajdata_indices = args.trajdata_indices,
                nsteps = args.max_path_length
                )
        env = JuliaNGSIMEnv(env_dict = env_dict,
                z_dim =args.z_dim, normalize_obs = True,
                prior_type = args.prior_type,
                n_egos = args.curr_start,
                end_on_failure = bool(args.end_on_failure))
    else:
        raise NotImplementedError
    env = TfEnv(env)
    if encoder is not None:
        env.wrapped_env.prior_f = encoder.predict
    return env

def _create_aux_networks(args, env):
    reward_trainer = TRAINER[args.reward_trainer]
    info_trainer = TRAINER[args.info_trainer]
    info_nonlin = NONLIN[args.info_nonlinearity]
    reward_nonlin = NONLIN[args.reward_nonlinearity]
    info_cell = CELL[args.info_cell]

    network_input_shape = env.spec.observation_space.flat_dim
    # create inverse dynamics model
    if args.invdyn_start_epoch > 0:
        invdyn_model = InverseDynamicsModel("invdyn", trainer, network_input_shape, env.action_dim,
                                      args.invdyn_hidden_sizes, info_nonlin, epochs=args.invdyn_epoch,
                                      active_epoch=args.invdyn_start_epoch, T=args.invdyn_temperature)
    else:
        invdyn_model = None

    # create variational posterior for latent code
    if args.use_infogail:
        if args.info_recur_dim > 0:
            InfoModel = RE_InfoModel
        else:
            InfoModel = FF_InfoModel

        info_model = InfoModel("info", info_trainer,
                                args.info_learning_rate,
                                obs_shape=network_input_shape,
                                act_dim=env.action_dim, z_dim = args.z_dim,
                                y_dim = len(args.domain_indices),
                                hidden_sizes=args.info_hidden_sizes,
                                hidden_nonlinearity=info_nonlin,
                                batch_size=args.info_batch_size,
                                epochs=args.info_epoch,
                                reg = args.info_reg,
                                ent = args.info_ent,
                                cnf = args.info_cnf,
                                # recurrent network arguments
                                max_path_length = args.max_path_length,
                                recur_dim = args.info_recur_dim,
                                decay_rate = args.info_decay_rate,
                                decay_step = args.info_decay_step,
                                drop_prob = args.info_drop_prob,
                                stochastic = args.info_stochastic,
                                cell = info_cell,
                                domain_indices = args.domain_indices,
                                include_cnf_in_reward=args.include_cnf_in_reward)
        if args.use_info_prior:
            env.wrapped_env.prior_f = info_model.predict
    else:
        info_model = None

    # create reward model
    reward_args = dict(
                    name="reward", trainer=reward_trainer,
                    learning_rate=args.reward_learning_rate,
                    obs_dim=network_input_shape,
                    act_dim=env.action_dim,
                    hidden_sizes=args.reward_hidden_sizes,
                    hidden_nonlinearity=reward_nonlin,
                    batch_size=args.reward_batch_size,
                    drop_prob=0.0, wgan=bool(args.wgan),
                    clip=args.clip, epochs=args.reward_epoch,
                    treat_z="ignore",
                    z_dim=args.z_dim,
                    z_idx=None,
                    z_hidden_sizes=None
                    )

    if args.model_all:
        assert args.domain_indices == [0]
        RewardCls = FactoredRewardModel
        reward_args["n_agents"] = args.curr_start
    else:
        RewardCls = RewardModel

    reward_model = RewardCls(**reward_args)

    return invdyn_model, reward_model, info_model, env

def _create_policy(args, env_spec):
    pi_nonlin = NONLIN[args.policy_nonlinearity]
    obs_dim = env_spec.observation_space.flat_dim
    action_dim = env_spec.action_space.flat_dim
    pi_args = dict(
            name="policy/mean_network",
            input_shape=(obs_dim,),
            output_dim=action_dim,
            hidden_sizes=args.policy_hidden_sizes,
            hidden_nonlinearity=pi_nonlin,
            output_nonlinearity=None
            )
    if vars(args).get("model_all",False):
        PiMeanCls = MLP
        PiCls = FactoredMLPPolicy
        pi_mean_args = dict(
                    name="policy/mean_network",
                    input_shape=(obs_dim,),
                    output_dim=action_dim,
                    hidden_sizes=args.policy_hidden_sizes,
                    hidden_nonlinearity=pi_nonlin,
                    output_nonlinearity=None
                    )
        pi_args['n_agents'] = args.curr_start

    else:
        if args.use_infogail:
            PiMeanCls = MergeMLP
            pi_mean_args = dict(
                        name="policy/mean_network",
                        merge=args.policy_merge,
                        input_shape=(obs_dim,),
                        output_dim=action_dim,
                        hidden_sizes=args.policy_hidden_sizes,
                        z_dim=args.z_dim, z_idx=args.z_policy_merge_idx,
                        z_hidden_sizes=args.z_policy_hidden_sizes,
                        hidden_nonlinearity=pi_nonlin,
                        output_nonlinearity=None
                        )
        else:
            PiMeanCls = MLP
            pi_mean_args = dict(
                        name="policy/mean_network",
                        input_shape=(obs_dim,),
                        output_dim=action_dim,
                        hidden_sizes=args.policy_hidden_sizes,
                        hidden_nonlinearity=pi_nonlin,
                        output_nonlinearity=None
                        )
        PiCls = PI_CELL[vars(args).get("policy_cell","none")]

    # create mean and std network
    mean_network = PiMeanCls(**pi_mean_args)
    if vars(args).get("adaptive_std",False):
        pi_mean_args["name"] = "policy/std_network"
        std_network = PiMeanCls(**pi_mean_args)
    else:
        std_network = None

    # create the policy instance
    policy = PiCls(
        name="policy",
        env_spec=env_spec,
        mean_network=mean_network,
        std_network=std_network,
        hidden_sizes=tuple(args.policy_hidden_sizes),
        hidden_nonlinearity=pi_nonlin,
        hidden_dim=args.policy_recur_dim
    )

    if args.bc_init:
        variables = sorted([(v.name, v) for v in policy.get_params()])
        with h5py.File("../{}".format(policy_paths["VAE"],"r")) as hf:
            L = [(n.replace("w","W"), v[...]) for n, v in hf["policy"].items() if ":0" in n]
            a_log = [L.pop(0)]
            sL = sorted(L)
            Z = [x for tup in zip(L[len(L)/2:],L[:len(L)/2]) for x in tup]
            weights = Z + a_log

        assert len(variables) == len(weights)
        init_ops = [tf.assign(v[-1],w[-1]) for v, w in zip(variables, weights)]
    else:
        init_ops = None
    return policy, init_ops

