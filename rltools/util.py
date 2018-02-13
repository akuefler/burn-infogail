from __future__ import print_function

import errno
import os
import timeit

import h5py
import numpy as np
from colorama import Fore, Style

from gym.spaces import Discrete, Box

class Timer(object):

    def __enter__(self):
        self.t_start = timeit.default_timer()
        return self

    def __exit__(self, _1, _2, _3):
        self.t_end = timeit.default_timer()
        self.dt = self.t_end - self.t_start

def nan_stack_ragged_array(X, mpl):
    Y = []
    len_B = []
    for x in X:
        try:
            t, d = x.shape
            new_shape = (mpl-t,d)
        except:
            t = x.shape[0]
            new_shape = (mpl-t,)
        len_B.append(t)
        if t < mpl:
            x = np.concatenate([x, np.ones(new_shape) * np.nan], axis=0)
        Y.append(x[None,...])
    return np.row_stack(Y), np.array(len_B)

def load_trajs(filename, limit_trajs, domain_indices= None, swap= False):
    # Load expert data
    stats = {}
    states = {}
    cls_data = {}
    with h5py.File(filename, 'r') as f:
        # Read data as written by scripts/format_data.py (openai format)
        if swap:
            obs= np.array(f['obs_B_T_Do']).T
            act= np.array(f['a_B_T_Da']).T
            #rew= np.array(f['r_B_T']).T
            lng= np.array(f['len_B']).T
        else:
            obs= np.array(f['obs_B_T_Do'])
            act= np.array(f['a_B_T_Da'])
            #rew= np.array(f['r_B_T'])
            lng= np.array(f['len_B'])

        # attempt to load VAE params,
        # if they exist
        z_data = {}

        B, T, Do = obs.shape
        keep = np.ones(B).astype('bool')

        # domain labels
        try:
            dom_B = np.array(f['dom_B'])
            if domain_indices is not None:
                keep = np.array([k in domain_indices for k in dom_B]).astype('bool')

            cls_data["dom_B"] = dom_B[keep]
        except KeyError:
            pass
        cls_data['keep'] = keep

        # note : after removing dom, change indices
        try:
            states['met'] = np.array(f['met_B_T_Dm'])[keep]
        except KeyError:
            pass
        try:
            z_data['z_mean'] = np.array(f['zmean_B_Dz'])[keep]
            z_data['z_logstd'] = np.array(f['zlogstd_B_Dz'])[keep]
        except KeyError:
            pass
        try:
            z_data['z_mean'] = np.array(f['zmean_B_T_Dz'])[keep]
            z_data['z_logstd'] = np.array(f['zstd_B_T_Dz'])[keep]
        except KeyError:
            print("skipped Z data")
        try:
            cls_data['cls_B'] = np.array(f['cls_B'])[keep]
        except KeyError:
            pass

        if 'state' in f.keys():
            for key, val in f['state/measure'].items():
                states["state/{}".format(key)] = val[...][keep]

        # attempt to load normalization params,
        # if they exist
        try:
            obs_mean = np.array(f['obs_mean'])
            obs_var = np.array(f['obs_var'])

            stats['obs_mean'] = obs_mean
            stats['obs_var'] = obs_var
        except KeyError:
            pass

        full_dset_size = obs.shape[0]
        dset_size = min(full_dset_size, limit_trajs) if limit_trajs is not None else full_dset_size

        exobs_B_T_Do = obs[:dset_size,...][...][keep]
        exa_B_T_Da = act[:dset_size,...][...][keep]
        exlen_B = lng[:dset_size,...][...][keep]

    data={'obs':exobs_B_T_Do,
          'act' : exa_B_T_Da,
          #'exr_B_T' : exr_B_T,
          'exlen_B' : exlen_B,
          #'interval':interval
          }
    data.update(z_data)
    data.update(cls_data)
    data.update(states)
    return data, stats

def prepare_trajs(exobs_B_T_Do, exa_B_T_Da, exlen_B, data_subsamp_freq= 1, labeller= None):
    print('exlen_B inside: %i'%exlen_B.shape[0])
    exlen_B = exlen_B.astype("int32")

    start_times_B = np.random.RandomState(0).randint(0, data_subsamp_freq, size=exlen_B.shape[0])
    exobs_Bstacked_Do = np.concatenate(
        [exobs_B_T_Do[i,start_times_B[i]:l:data_subsamp_freq,:] for i, l in enumerate(exlen_B)],
        axis=0)
    exa_Bstacked_Da = np.concatenate(
        [exa_B_T_Da[i,start_times_B[i]:l:data_subsamp_freq,:] for i, l in enumerate(exlen_B)],
        axis=0)

    assert exobs_Bstacked_Do.shape[0] == exa_Bstacked_Da.shape[0]

    data={'exobs_Bstacked_Do':exobs_Bstacked_Do,
          'exa_Bstacked_Da' : exa_Bstacked_Da}

    return data

