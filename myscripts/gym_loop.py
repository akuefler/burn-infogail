import gym
#from gym.envs.box2d import CarRacing
from rllab.envs.racing import CarRacing
from gym.envs.classic_control import PendulumEnv
from rllab.envs.parameterized_pendulum import ParameterizedPendulumEnv
from rllab.envs.parameterized_ll import *

import numpy as np
import matplotlib.pyplot as plt

import argparse

import time

parser = argparse.ArgumentParser()
# Racing Simulator params
parser.add_argument("--environment",type=str,default='Racing-State')
parser.add_argument("--track_turn_rate",type=float,default=0.31)
parser.add_argument("--track_width",type=int,default=40)
parser.add_argument("--friction_limit",type=int,default=1000000)
parser.add_argument("--wheel_moment_of_inertia",type=int,default=4000)
parser.add_argument("--engine_power",type=int,default=100000000)
parser.add_argument("--brake_force",type=int,default=15)
parser.add_argument("--road_friction",type=float,default=0.9)
parser.add_argument("--grass_friction",type=float,default=0.6)
# Racing Features
#parser.add_argument("--features",type=str,nargs="+",default=["pos","vel","abs","hull_ang","wheel_ang",
#                                                             "reward","hull_ang_vel","speed","xy_dist_from_road"])

#parser.add_argument("--features",type=str,nargs="+",default=["vel","hull_ang","wheel_ang",
#                                                             "hull_ang_vel","speed","xy_dist_from_road"])
#parser.add_argument("--features",type=str,nargs="+",default=["vel","speed","hull_ang","xy_dist_from_road","curves","on_grass",
#    "lane_offset","heading_angle"])


parser.add_argument("--features",type=str,nargs="+",default=["vel", "curves", "on_grass", "lane_offset", "heading_angle"])

args = parser.parse_args()

env = CarRacing(mode="state",features=args.features,
                track_turn_rate=args.track_turn_rate,
                track_width=args.track_width,
                friction_limit=args.friction_limit,
                wheel_moment_of_inertia=args.wheel_moment_of_inertia,
                engine_power=args.engine_power, brake_force=args.brake_force,
                road_friction=args.road_friction, grass_friction=args.grass_friction
                )
#env = PendulumEnv()
"""
g is gravity? Low g leads to smoother, floatier swings. High g leads to
tight, quick swings.

m is mass of rod

l is length of rod?
"""
#env = ParameterizedPendulumEnv(max_speed=10., max_torque=4., g = 1.10 * 10., l = 1, m = 1.)
#env._seed(seed=456)

#env = LunarLanderContinuous(main_engine_power=13.0, side_engine_power=0.6,
                 #side_engine_height = 14.0, side_engine_away=12.0,
                 #lander_density=5.0)
#env = LunarLanderContinuous(main_engine_power=7.0, side_engine_power=0.8,
#                 side_engine_height = 14.0, side_engine_away=12.0,
#                 lander_density=5.0)

E = 10
T = 1000
plt.ion()

#f, (axs) = plt.subplots(3)
plt.ion()
f, ax = plt.subplots(1,1)
for e in range(E):
    o = env.reset()
    for t in range(T):

        a = env.action_space.sample()
        a = np.array([0.0,1.0,0.0]) * 100.

        o, r, done, info = env.step(a)

        env.render()
        if True:
            plt.cla()
            ax.bar(np.arange(len(o)), o)
            ax.set_ylim(-1.5,1.5)
            plt.show()
            plt.pause(0.001)
        time.sleep(0.05)

        if done:
            break
