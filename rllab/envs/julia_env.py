import julia
import gym

import numpy as np
import matplotlib.pyplot as plt

from rllab.config import *
from gym.spaces import Box

from trn.config import policy_paths, normalizer_paths
from trn.config import expert_data_paths
#PROJECT_PATH = osp.abspath(osp.join(osp.dirname(__file__), '..'))

import rltools.util
import h5py

import copy

class JuliaEnv(gym.Env):
    """
    base class for both Julia environments
    """
    def sample_prior(self, prime_obs, prime_act):
        if self.prior_f is not None:
            prime_obs = (prime_obs - self.obs_mean) / self.obs_std
            prime_act = (prime_act - self.act_mean) / self.act_std


            prime_obs = \
                np.column_stack([prime_obs,np.zeros_like(prime_obs)[:,:self.z_dim]])
            eos = np.array([prime_obs.shape[0]])
            z, _ = self.prior_f(prime_obs[None,...], prime_act[None,...], EOS = eos)
            if self.prior_type == "discrete":
                z = np.eye(self.z_dim)[z[0]]
            else:
                z = np.squeeze(z)
            z_mean, z_std = z, z

        else:
            if self.prior_type == "discrete":
                z = np.eye(self.z_dim)[np.random.randint(self.z_dim)]
                z_mean, z_std = z, z # no mean, std for discrete codes.
            else:
                n = self.z_mean.shape[0]
                z_mean = self.z_mean[np.random.randint(n)]
                z_std = self.z_std[np.random.randint(n)]
                z = np.random.normal(z_mean, z_std)
        return z, z_mean, z_std

    def copy_simparams(self):
        return self.j.copy_simparams(self.simparams)

    def render(self, close= False):
        #IMG = np.zeros((2000,1200)).astype("uint32")
        IMG = np.zeros((500,500)).astype("uint32")
        IMG = self.j.retrieve_image(self.simparams, IMG)
        self.ax.cla()
        self.ax.matshow(IMG, cmap=plt.cm.Paired)
        plt.show()

        plt.pause(0.001)

    def _observe(self):
        state = self.j.observe_state(self.simparams)

        obs = np.array(self.j.observe(self.simparams))

        obs_mean = self.obs_mean
        obs_std = self.obs_std

        obs = (obs - obs_mean) / obs_std

        return np.concatenate([obs,self.z]), state

    def reset(self, seed=-1):
        self.j.reset(self.simparams, self.n_egos, seed)
        #self.initial_simparams = self.j.copy_simparams(self.simparams)

        if self.z_dim > 0:
            prime = self.j.retrieve_prime(self.simparams)
            prime_obs = np.row_stack([prime["obs"+str(i + 1)] for i in
                range(0,len(prime.keys())/2)])
            prime_act = np.row_stack([prime["act"+str(i + 1)] for i in
                range(0,len(prime.keys())/2)])

            self.z, self.curr_z_mean, self.curr_z_std = self.sample_prior(
                    prime_obs, prime_act
                    )

        observation, _ = self._observe()
        return observation

    def step(self, action):
        if len(action) > 2:
            action = np.concatenate([
                np.clip(act, *self.j.action_space_bounds(self.simparams))
                for act in np.split(action,len(action)/2)
                ])
        else:
            action = np.clip(action, *self.j.action_space_bounds(self.simparams))

        info = {}
        reward = 0.0
        self.j.step(self.simparams, action)
        #observation = self._observe(self.j.observe(self.simparams), action)
        observation, state = self._observe()
        done = self.j.isdone(self.simparams)
        info = self.j.get_info(self.simparams)

        # use collisions / off road to assign reward.
        rewards = -1. * (info["collision"] + info["offroad"] + info["reverse"])
        info['rewards'] = rewards

        reward = np.sum(rewards)
        if reward < 0.0 and self.end_on_failure:
            done = True
        # add information about the prior to info
        if self.z_dim > 0:
            info.update({"z":self.z, "z_mean":self.curr_z_mean, "z_std":self.curr_z_std})

        info["state"] = state

        return observation, reward, done, info

    @property
    def obs_dim(self):
        lo, _ = self.j.observation_space_bounds(self.simparams)
        #return len(lo) + (int(self.n_egos > 1) * self.n_egos *
        #        int(self.index_features)
        #        )
        return len(lo)

    @property
    def act_dim(self):
        lo, _ = self.j.action_space_bounds(self.simparams)
        #return len(lo) + (int(self.n_egos > 1) * self.n_egos)
        return len(lo)

    @property
    def observation_space(self):
        lo, hi = self.j.observation_space_bounds(self.simparams)
        lo = np.concatenate([lo, float("-inf") * np.ones(self.z_dim)])
        hi = np.concatenate([hi, float("inf") * np.ones(self.z_dim)])
#        if (self.n_egos > 1):
#            if self.index_features:
#                lo = np.concatenate([lo, np.zeros(self.n_egos)])
#                hi = np.concatenate([hi, np.zeros(self.n_egos)])
#
#            lo = np.repeat(lo,self.n_egos)
#            hi = np.repeat(hi,self.n_egos)

        return Box(lo,hi)

    @property
    def action_space(self):
        lo, hi = self.j.action_space_bounds(self.simparams)
#        if self.n_egos > 1:
#            lo = np.repeat(lo,self.n_egos)
#            hi = np.repeat(hi,self.n_egos)

        return Box(lo,hi)

    @property
    def prior_f(self):
        return self._prior_f

    @prior_f.setter
    def prior_f(self, f):
        self._prior_f = f

    @property
    def n_egos(self):
        return self._n_egos

    @n_egos.setter
    def n_egos(self, v):
        #n_vehicles = self.j.get_n_vehicles(self.simparams)
        self._n_egos = v


class JuliaTrackEnv(JuliaEnv):
    def __init__(self, env_dict = {}, n_egos = 1, z_dim = 0, normalize_obs = True, prior_type = "discrete",
            end_on_failure = True, index_features = False):
        self.j = julia.Julia()
        include_statement = "include(\"{}\")".format(AUTO2D_PATH)
        self.j.eval(include_statement)
        self.j.using("Auto2D")
        self.simparams = self.j.gen_simparams(env_dict)

        self.mix_class = env_dict["mix_class"]
        self.model_all = env_dict["model_all"]
        self.domain_indices = env_dict["domain_indices"]

        self.index_features = index_features

        self.z_dim = z_dim

        self._prior_f = None
        self._n_egos = n_egos

        self.prior_type = prior_type
        self.end_on_failure = end_on_failure
        # pull observation mean and std for normalization
        with h5py.File("{}/{}".format(PROJECT_PATH,normalizer_paths["ORIG"]),"r") as hf:
            self.obs_mean = \
                    np.repeat(hf['obs_mean'][...],self.n_egos)
            self.obs_std = \
                    np.repeat(hf['obs_std'][...],self.n_egos)

            self.act_mean = \
                    np.repeat(hf['act_mean'][...],self.n_egos)
            self.act_std = \
                    np.repeat(hf['act_std'][...], self.n_egos)

        if not normalize_obs:
            self.obs_mean = np.zeros_like(self.obs_mean)
            self.obs_std = np.ones_like(self.obs_std)

        # controls latent label
        if self.z_dim > 0:
            if prior_type == "standard": # standard gaussian prior
                self.z_mean = np.zeros(self.z_dim)[None,...]
                self.z_std = np.ones(self.z_dim)[None,...]
            elif prior_type == "discrete":
                self.z_mean, self.z_std = np.zeros(self.z_dim),np.zeros(self.z_dim)
        else:
            self.z = []

        obsdim = self.j.get_obsdim(self.simparams)
        _, self.ax = plt.subplots(1,1)
        plt.ion()

    def save_gif(self, simparams, actions, filename, truth_simparams = None):
        # need to ensure simparams actually matches the initial conditions.
        # egos chosen, etc ... 
        #self.j.reel_drive(filename+".gif", actions, self.initial_simparams)

        self.j.reel_drive(filename+".gif", actions, simparams)
        if truth_simparams is not None:
            self.j.reel_drive(filename+"_TRUTH.gif", truth_simparams)


class JuliaNGSIMEnv(JuliaEnv):
    def __init__(self, env_dict = {}, z_dim = 0, normalize_obs = True, prior_type = "discrete",
            end_on_failure = True):
        self.j = julia.Julia()
        self.j.eval("include(\"{}\")".format(AUTO2D_PATH))
        self.j.using("Auto2D")

        #nsteps = get(args, "nsteps", 20) #ASSUME : this right?
        self.simparams = self.j.gen_simparams(1,env_dict)

        self.prior_type = prior_type
        self.end_on_failure = end_on_failure

        # pull observation mean and std for normalization
        with h5py.File("{}/{}".format(DATA_DIR,normalizer_paths["JNGSIM"]),"r") as hf:
            self.obs_mean = hf['obs_mean'][...]
            self.obs_std = hf['obs_std'][...]

            self.act_mean = hf['act_mean'][...]
            self.act_std = hf['act_std'][...]

        if not normalize_obs:
            self.obs_mean = np.zeros_like(self.obs_mean)
            self.obs_std = np.ones_like(self.obs_std)

        # controls latent label
        self.z_dim = z_dim
        if prior_type == "standard": # standard gaussian prior
            self.z_mean = np.zeros(self.z_dim)[None,...]
            self.z_std = np.ones(self.z_dim)[None,...]
        elif prior_type == "discrete":
            self.z_mean, self.z_std = np.zeros(self.z_dim),np.zeros(self.z_dim)
        else:
            raise NotImplementedError

        obsdim = self.j.get_obsdim(self.simparams)
        _, self.ax = plt.subplots(1,1)
        plt.ion()

if __name__ == "__main__":
    #env = JuliaNGSIMEnv(z_dim = 5, env_dict = {"trajdata_indices":[1]})
    env_dict = {"domain_indices":[1], "mix_class":True,
            "model_all":False, "use_valid":False}
    env = JuliaTrackEnv(env_dict, z_dim = 0, end_on_failure = False, n_egos = 1)
    E = 1
    #T = 1000
    T = 300

    for e in range(E):
        o = env.reset()
        A = []
        for t in range(T):
            print(t)
            #a = np.random.normal(np.ones(2),np.ones(2)).astype('float64')
            #a = np.array([-3.,0.] + [100.,1.])
            #a = np.concatenate([a]*env.n_egos)
            a = np.array([-0.01, 1.0]) * 100.
            #print("Epoch: {} ; Time: {} ; Action: {}".format(e,t,a))
            #a = np.zeros(2)
            o_, r, d, info = env.step(a)

            A.append(a)
            if d: break
            env.render()

        #env.save_gif(env.simparams, np.column_stack(A), "./output0")
        #env.save_gif(env.initial_simparams, np.column_stack(A), "./output1")
        #        env.wrapped_env.save_gif(trajbatch['actions'].T,
        #                "{}/sample{}".format(config.PROJECT_PATH + "/gifs", i))


    halt = True

