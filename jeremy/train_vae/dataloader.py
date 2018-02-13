import collections
import cPickle
import h5py
import math
import numpy as np
import os
import random

from rllab.config import EXPERT_PATH
#name = "{}/{}/{}".format(EXPERT_PATH,["juliaTrack_single","juliaTrack_mix"][args.mix],"expert_trajs.h5")

# Class to load and preprocess data
class DataLoader():
    def __init__(self, batch_size, seq_length, mix, domains=[]):
        self.batch_size = batch_size
        self.seq_length = seq_length

        self.domains = domains

        self.data_path = "{}/{}/".format(EXPERT_PATH,["juliaTrack_single","juliaTrack_mix"][mix])

        print "loading data..."
        self._load_data()

        print 'creating splits...'
        self._create_split()

        #print 'shifting/scaling data...'
        #self._shift_scale()

    def _trim_data(self, full_s, full_a, intervals):
        # Python indexing; find bounds on data given seq_length
        intervals -= 1
        lengths = np.floor(np.diff(np.append(intervals, len(full_s)-1))/self.seq_length)*self.seq_length
        intervals = np.vstack((intervals, intervals + lengths)).T.astype(int)
        ret_bounds = np.insert(np.cumsum(lengths), 0, 0.).astype(int)

        # Remove states that don't fit due to value of seq_length
        s = np.zeros((int(sum(lengths)), full_s.shape[1]))
        for i in xrange(len(ret_bounds)-1):
            s[ret_bounds[i]:ret_bounds[i+1]] = full_s[intervals[i, 0]:intervals[i, 1]]
        s = np.reshape(s, (-1, self.seq_length, full_s.shape[1]))

        # Remove actions that don't fit due to value of seq_length
        a = np.zeros((int(sum(lengths)), full_a.shape[1]))
        for i in xrange(len(ret_bounds)-1):
            a[ret_bounds[i]:ret_bounds[i+1]] = full_a[intervals[i, 0]:intervals[i, 1]]
        a = np.reshape(a, (-1, self.seq_length, full_a.shape[1]))

        return s, a

    def _load_and_format_data(self, name):
        with h5py.File("{}/{}/expert_trajs.h5".format(self.data_path,name),"r") as hf:
            s = hf['obs_B_T_Do'][...]
            a = hf['a_B_T_Da'][...]
            c = hf['cls_B'][...]
            d = hf['dom_B'][...]

            keep = np.array([k in self.domains for k in d]).astype('bool')
            s = s[keep]
            a = a[keep]
            c = c[keep]

        # Make sure batch_size divides into num of examples 
        B, T, Do = s.shape
        c = (c[...,None].repeat(T,axis=1))[...,None]
        s = np.concatenate([s,c],axis=-1)

        print("len(s): {}".format(len(s)))
        assert len(s) % self.batch_size == 0
        s = np.reshape(s, (-1, self.batch_size, self.seq_length, s.shape[2]))
        a = np.reshape(a, (-1, self.batch_size, self.seq_length, a.shape[2]))

        # Now separate states and classes
        c = s[:, :, :, 51]
        s = s[:, :, :, :51]

        # Create batch_dict
        self.batch_dict = {}
        self.batch_dict["states"] = np.zeros((self.batch_size, self.seq_length, s.shape[2]))
        self.batch_dict["actions"] = np.zeros((self.batch_size, self.seq_length, a.shape[2]))
        self.batch_dict["classes"] = np.zeros((self.batch_size, self.seq_length))

        return s, c, a

    def _load_data(self):

        s, c, a = self._load_and_format_data("train")
        self.s = s
        self.c = c
        self.a = a

        s, c, a = self._load_and_format_data("valid")
        self.s_v = s
        self.c_v = c
        self.a_v = a

    # Separate data into train/validation sets
    def _create_split(self):

        # compute number of batches
        self.n_batches = len(self.s)
        self.n_batches_val = len(self.s_v)
        self.n_batches_train = self.n_batches - self.n_batches_val

        print 'num training batches: ', self.n_batches_train
        print 'num validation batches: ', self.n_batches_val

        self.reset_batchptr_train()
        self.reset_batchptr_val()

    # Sample a new batch of data
    def next_batch_train(self):
        # Extract next batch
        batch_index = self.batch_permuation_train[self.batchptr_train]
        self.batch_dict["states"] = self.s[batch_index]
        self.batch_dict["actions"] = self.a[batch_index]
        self.batch_dict["classes"] = self.c[batch_index]

        # Update pointer
        self.batchptr_train += 1
        return self.batch_dict

    # Return to first batch in train set
    def reset_batchptr_train(self):
        self.batch_permuation_train = np.random.permutation(self.n_batches_train)
        self.batchptr_train = 0

    # Return next batch of data in validation set
    def next_batch_val(self):
        # Extract next validation batch
        batch_index = self.batchptr_val
        self.batch_dict["states"] = self.s_v[batch_index]
        self.batch_dict["actions"] = self.a_v[batch_index]
        self.batch_dict["classes"] = self.c_v[batch_index]

        # Update pointer
        self.batchptr_val += 1
        return self.batch_dict

    # Return to first batch in validation set
    def reset_batchptr_val(self):
        self.batchptr_val = 0

    # Sample a new batch of data from passive set
    def next_batch_pass(self):
        # Extract next batch
        self.batch_dict["states"] = self.s_pass[self.batchptr_pass]
        self.batch_dict["actions"] = self.a_pass[self.batchptr_pass]

        # Update pointer
        self.batchptr_pass += 1
        return self.batch_dict

    # Sample a new batch of data from passive set
    def next_batch_agg(self):
        # Extract next batch
        self.batch_dict["states"] = self.s_agg[self.batchptr_agg]
        self.batch_dict["actions"] = self.a_agg[self.batchptr_agg]

        # Update pointer
        self.batchptr_agg += 1
        return self.batch_dict

    # Sample a new batch of data from passive set
    def next_batch_med1(self):
        # Extract next batch
        self.batch_dict["states"] = self.s_med1[self.batchptr_med1]
        self.batch_dict["actions"] = self.a_med1[self.batchptr_med1]

        # Update pointer
        self.batchptr_med1 += 1
        return self.batch_dict

    # Sample a new batch of data from passive set
    def next_batch_med2(self):
        # Extract next batch
        self.batch_dict["states"] = self.s_med2[self.batchptr_med2]
        self.batch_dict["actions"] = self.a_med2[self.batchptr_med2]

        # Update pointer
        self.batchptr_med2 += 1
        return self.batch_dict

