import argparse
import calendar
import os
import os.path as osp
from rllab import config

import gym
import tensorflow as tf
import numpy as np

from tf_rllab.algos.gail import GAIL
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.box2d.cartpole_env import CartpoleEnv
from rllab.envs.normalized_env import normalize
from tf_rllab.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer
from tf_rllab.optimizers.conjugate_gradient_optimizer import FiniteDifferenceHvp
from tf_rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from tf_rllab.core.network import RewardMLP
from tf_rllab.core.network import AdaptiveRewardLSTM, InverseDynamicsModel, ConvNetwork
from rllab.envs.tf_env import TfEnv

from rllab.envs.racing import CarRacing
from rllab.envs.parameterized_pendulum import ParameterizedPendulumEnv

from tf_rllab import RLLabRunner

from rllab.config_personal import expert_trajs_path, model_path

from rllab.envs.gym_env import GymEnv

from tf_rllab.baselines.gaussian_conv_baseline import GaussianConvBaseline

import rltools.util

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

parser.add_argument('--save_policy',type=int,default=0)
parser.add_argument('--itr_threshold',type=int,default=100)
parser.add_argument('--reward_threshold',type=int,default=-1000.)

# general params
parser.add_argument("--n_itr",type=int,default=750)
parser.add_argument("--max_traj_len",type=int,default=150)
parser.add_argument("--fix_perfect_weights",type=int,default=0)

parser.add_argument("--discount",type=int,default=0.99)

parser.add_argument("--normalize",type=int,default=0)

# architecture params
parser.add_argument("--nonlinearity",type=str,default="relu")
parser.add_argument("--policy_hidden_sizes",type=int,nargs="+",default=[64,32,32,16])
parser.add_argument("--confus_hidden_sizes",type=int,nargs="+",default=[32,16,1])
parser.add_argument("--transf_hidden_sizes",type=int,nargs="+",default=[32,16,16])
parser.add_argument("--reward_hidden_sizes",type=int,nargs="+",default=[64, 32,16,1])

parser.add_argument("--recurrent_units",type=int,default=2)
parser.add_argument("--cell_type",type=str,default='mlp')

parser.add_argument("--transform_actions",type=int,default=0)

parser.add_argument("--baseline_type",type=str,default="linear")

# network params
parser.add_argument("--r_drop_prob",type=float,default=0.0)
parser.add_argument("--t_drop_prob",type=float,default=0.0)
parser.add_argument("--c_drop_prob",type=float,default=0.0)

parser.add_argument("--trainer",type=str,default="rmsprop")

# convolutional params
parser.add_argument("--conv_filters",type=int,nargs="+",default=[32,64,64]) 
parser.add_argument("--conv_filter_sizes",type=int,nargs="+",default=[8,6,4]) 
parser.add_argument("--conv_strides",type=int,nargs="+",default=[4,3,2])

# gail params
parser.add_argument("--trpo_step_size",type=float,default=0.01)
parser.add_argument("--trpo_batch_size",type=int,default=50 * 100)
parser.add_argument("--gail_batch_size",type=int,default=50)

parser.add_argument("--learning_rate",type=float,default=0.00005)
parser.add_argument("--decay_rate",type=float,default=0.99)
parser.add_argument("--decay_steps",type=int,default=20)

parser.add_argument("--disc_steps",type=int,default=5)

# inverse dynamics model
parser.add_argument("--invdyn_hidden_sizes",type=int,nargs='+',default=[32,32])
parser.add_argument("--invdyn_epoch",type=int,default=0)

# disable features
parser.add_argument("--disable_policy",type=int,default=0)
parser.add_argument("--disable_flip_gradient",type=int,default=0)
parser.add_argument("--flip_reward",type=int,default=0)

# transformer params
parser.add_argument("--t_w_trainable",type=int,default=1)
parser.add_argument("--t_b_trainable",type=int,default=1)
parser.add_argument("--d_trainable",type=int,default=1)
parser.add_argument("--cost_weight",type=float,default=1.0)

parser.add_argument("--share_weights",type=int,default=1)

# GAN hack params
parser.add_argument("--expert_obs_noise_scale",type=float,default=0.0)
parser.add_argument("--wgan",type=int,default=1)

parser.add_argument("--conf_clip",type=float,default=1000.)
#parser.add_argument("--disc_clip",type=float,default=0.01)
#parser.add_argument("--transf_clip",type=float,default=0.01)
parser.add_argument("--disc_clip",type=float,default=1000.)
parser.add_argument("--transf_clip",type=float,default=1000.)

parser.add_argument("--environment",type=str,default='Racing-State')
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
parser.add_argument("--features",type=str,nargs="+",
                    default=["pos","vel","abs","hull_ang","wheel_ang",
                             "reward","hull_ang_vel","speed","xy_dist_from_road"])
#################
# PENDULUM PARAMS
#################
parser.add_argument("--max_speed",type=float,default=10.)
parser.add_argument("--max_torque",type=float,default=4.)
parser.add_argument("--g",type=float,default=10.)
parser.add_argument("--m",type=float,default=1.)
parser.add_argument("--l",type=float,default=1.)

# misc params
parser.add_argument("--debug",type=int,default=0)
parser.add_argument("--seed",type=int,default=456)
parser.add_argument("--expert_data_path",type=str,default="expert_trajs/racing/Racing-State-0")

args = parser.parse_args()

if __name__ == "__main__":
    np.random.seed(args.seed)
    tf.set_random_seed(args.seed)

    #env = TfEnv(normalize(CartpoleEnv())) ## normalize or not ?
    if args.environment == "CartPole":
        env = TfEnv(CartpoleEnv())
    elif args.environment == "Pendulum":
        env = gym.make("Pendulum-v0")
        env = TfEnv(env)
        #t_hidden_sizes = ()
    elif args.environment == "NoisyPendulum":
        gym.envs.register(
            id="NoisyPendulum-v0",
            entry_point='rllab.envs.target_env:NoisyPendulum',
            timestep_limit=999,
            reward_threshold=195.0,
        )
        env = TfEnv(GymEnv("NoisyPendulum-v0"))
    elif args.environment in ["Racing-State", "Racing-State-Action"]:
        #env = TfEnv(CarRacing(mode="pixels"))
        if args.environment == "Racing-State":
            mode = 'state'
        elif args.environment == 'Racing-State-Action':
            mode = 'state_action'
        env = CarRacing(mode=mode,features=args.features,
                        track_turn_rate=args.track_turn_rate, 
                        track_width=args.track_width,
                        friction_limit=args.friction_limit, 
                        wheel_moment_of_inertia=args.wheel_moment_of_inertia, 
                        engine_power=args.engine_power, brake_force=args.brake_force, 
                        road_friction=args.road_friction, grass_friction=args.grass_friction
                        )
        env = TfEnv(env)
    elif args.environment == 'PP':
        env = ParameterizedPendulumEnv(max_speed=args.max_speed, max_torque=args.max_torque,
                                       g=args.g, m=args.m, l=args.l)
        env = TfEnv(env)
        
    nonlinearity = {"tanh":tf.nn.tanh,"relu":tf.nn.relu,"elu":tf.nn.elu}[args.nonlinearity]
    cell = {"lstm": tf.nn.rnn_cell.BasicLSTMCell(args.recurrent_units, state_is_tuple= True),
             "gru": tf.nn.rnn_cell.GRUCell(args.recurrent_units),
             "rnn": tf.nn.rnn_cell.BasicRNNCell(args.recurrent_units),
             "mlp": None}[args.cell_type]
    trainer = {"adam":tf.train.AdamOptimizer,
               "sgd":tf.train.GradientDescentOptimizer,
               "rmsprop":tf.train.RMSPropOptimizer}[args.trainer]

    use_trajs = not (cell is None)
    
    init_ops = []
    expert_data, stats = rltools.util.load_trajs("../data/{}/expert_trajs.h5".format(args.expert_data_path), 12000, swap=False)    
    if args.normalize:
        env = TfEnv(normalize(env, initial_obs_mean=stats['obs_mean'],
                              initial_obs_var=stats['obs_std'] ** 2, normalize_obs=True))
    expert_data = {'obs':expert_data["exobs_B_T_Do"], 'act':expert_data["exa_B_T_Da"]}
        
    if not bool(cell is None):
        pass
    
    if len(env.observation_space.shape) > 2:
        # name, input_shape, output_dim
        mean_network = ConvNetwork("cnn", env.observation_space.shape, env.action_dim, 
                                  args.conv_filters, 
                                  args.conv_filter_sizes, 
                                  args.conv_strides, 
                                  ["VALID"] * len(args.conv_filters), 
                                  args.policy_hidden_sizes, 
                                  nonlinearity, 
                                  output_nonlinearity=None)
        policy = GaussianMLPPolicy(
            name="policy",
            env_spec=env.spec,
            mean_network=mean_network,
        )
    else:
        policy = GaussianMLPPolicy(
            name="policy",
            env_spec=env.spec,
            hidden_sizes=tuple(args.policy_hidden_sizes)
        )

    ex_env_obs_shape = (expert_data['obs'].shape[-1],)

    if use_trajs:
        max_step = args.max_traj_len
    else:
        max_step = 1
        
    #conv_params['hws'], conv_params['channels'], conv_params['strides'], conv_params['pads']
    conv_params = {'hws':args.conv_filter_sizes,'channels':args.conv_filters,
                   'strides':args.conv_strides,'pads':["VALID"]*len(args.conv_strides)}
    clip_weights = {'transf':args.transf_clip,'disc':args.disc_clip, 'conf':args.conf_clip}
    if args.invdyn_epoch > 0:
        invdyn_trainer = tf.train.AdamOptimizer()
        invdyn = InverseDynamicsModel("invdyn", invdyn_trainer, env.spec.observation_space.shape[-1], env.action_dim,
                                      args.invdyn_hidden_sizes, tf.nn.relu, active_epoch=args.invdyn_epoch)
    else:
        invdyn = None
    reward = AdaptiveRewardLSTM("reward", trainer, cell, max_step,
                                ex_env_obs_shape, env.action_dim,
                                env.spec.observation_space.shape, env.action_dim,
                                tuple(args.reward_hidden_sizes),
                                tuple(args.transf_hidden_sizes),
                                tuple(args.confus_hidden_sizes),
                                t_w_trainable= bool(args.t_w_trainable),
                                t_b_trainable= bool(args.t_b_trainable),
                                d_trainable = bool(args.d_trainable),
                                cost_weight=args.cost_weight,
                                hidden_nonlinearity= nonlinearity,
                                transform_actions=args.transform_actions,
                                disable_policy=args.disable_policy,
                                disable_flip_gradient=args.disable_flip_gradient,
                                d_drop_prob=args.r_drop_prob,
                                t_drop_prob=args.t_drop_prob,
                                c_drop_prob=args.c_drop_prob,
                                flip_reward=bool(args.flip_reward),
                                wgan=bool(args.wgan),
                                share_weights=args.share_weights,
                                clip_weights= clip_weights,
                                conv_params=conv_params)
    
    if args.baseline_type == "cnn":
        baseline = GaussianConvBaseline(env.spec,hidden_nonlinearity=nonlinearity,
                                        conv_filters=args.conv_filters,
                                        conv_filter_sizes=args.conv_filter_sizes,
                                        conv_strides=args.conv_strides,
                                        hidden_sizes=args.policy_hidden_sizes)
    elif args.baseline_type == "linear":
        baseline = LinearFeatureBaseline(env_spec=None)
    else:
        pass
    
    algo = GAIL(
	debug = args.debug,
        save_policy=bool(args.save_policy),
        itr_threshold = args.itr_threshold,
        reward_threshold = args.reward_threshold,
        env=env,
        policy=policy,
        baseline=baseline,
        reward=reward,
        invdyn=invdyn,        
        expert_data=expert_data,
        batch_size= args.trpo_batch_size,
        gail_batch_size=args.gail_batch_size,
        max_path_length=args.max_traj_len,
        n_itr=args.n_itr,
        discount= args.discount,
        step_size=args.trpo_step_size,
        force_batch_sampler= True,
        whole_paths= True,
        adam_steps= 1,
        decay_rate= args.decay_rate,
        decay_steps= args.decay_steps,
        disc_steps= args.disc_steps,
        init_ops = init_ops,
        fo_optimizer_cls=trainer,
        load_params_args = None,
        use_trajs = use_trajs,
        expert_obs_noise_scale = args.expert_obs_noise_scale,
        fo_optimizer_args= dict(learning_rate = args.learning_rate,
                                beta1 = 0.9,
                                beta2 = 0.99,
                                epsilon= 1e-8),
        optimizer=ConjugateGradientOptimizer(hvp_approach=FiniteDifferenceHvp(base_eps=1e-5))
    )
    
    # use date and time to create new logging directory for each run
    date= calendar.datetime.date.today().strftime('%y-%m-%d')
    if date not in os.listdir(model_path):
        os.mkdir(model_path+'/'+date)
    
    c = 0
    exp_name = args.environment + '-{}-'.format(args.exp_name) + str(c)
    
    while exp_name in os.listdir(model_path+'/'+date+'/'):
        c += 1
        exp_name = args.environment + '-{}-'.format(args.exp_name)+str(c)
    
    exp_dir = date+'/'+exp_name
    log_dir = osp.join(config.LOG_DIR, exp_dir)
    
    policy.set_log_dir(log_dir)
    
    runner = RLLabRunner(algo, args, exp_dir)
    runner.train()
