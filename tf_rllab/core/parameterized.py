from contextlib import contextmanager

from rllab.core.serializable import Serializable
from rllab.misc.tensor_utils import flatten_tensors, unflatten_tensors
import tensorflow as tf
import numpy as np

import h5py
import os

load_params = True

@contextmanager
def suppress_params_loading():
    global load_params
    load_params = False
    yield
    load_params = True


class Parameterized(object):
    def __init__(self):
        self._cached_params = {}
        self._cached_param_dtypes = {}
        self._cached_param_shapes = {}
        self._cached_assign_ops = {}
        self._cached_assign_placeholders = {}
        self.save_name = 'epochs'

    def get_params_internal(self, **tags):
        """
        Internal method to be implemented which does not perform caching
        """
        raise NotImplementedError

    def get_params(self, **tags):
        """
        Get the list of parameters, filtered by the provided tags.
        Some common tags include 'regularizable' and 'trainable'
        """
        tag_tuple = tuple(sorted(list(tags.items()), key=lambda x: x[0]))
        if tag_tuple not in self._cached_params:
            self._cached_params[tag_tuple] = self.get_params_internal(**tags)
        return self._cached_params[tag_tuple]

    def get_param_dtypes(self, **tags):
        tag_tuple = tuple(sorted(list(tags.items()), key=lambda x: x[0]))
        if tag_tuple not in self._cached_param_dtypes:
            params = self.get_params(**tags)
            param_values = tf.get_default_session().run(params)
            self._cached_param_dtypes[tag_tuple] = [val.dtype for val in param_values]
        return self._cached_param_dtypes[tag_tuple]

    def get_param_shapes(self, **tags):
        tag_tuple = tuple(sorted(list(tags.items()), key=lambda x: x[0]))
        if tag_tuple not in self._cached_param_shapes:
            params = self.get_params(**tags)
            param_values = tf.get_default_session().run(params)
            self._cached_param_shapes[tag_tuple] = [val.shape for val in param_values]
        return self._cached_param_shapes[tag_tuple]

    def get_param_values(self, **tags):
        params = self.get_params(**tags)
        param_values = tf.get_default_session().run(params)
        return flatten_tensors(param_values)

    def set_param_values(self, flattened_params, **tags):
        debug = tags.pop("debug", False)
        param_values = unflatten_tensors(
            flattened_params, self.get_param_shapes(**tags))
        ops = []
        feed_dict = dict()
        for param, dtype, value in zip(
                self.get_params(**tags),
                self.get_param_dtypes(**tags),
                param_values):
            if param not in self._cached_assign_ops:
                assign_placeholder = tf.placeholder(dtype=param.dtype.base_dtype)
                assign_op = tf.assign(param, assign_placeholder)
                self._cached_assign_ops[param] = assign_op
                self._cached_assign_placeholders[param] = assign_placeholder
            ops.append(self._cached_assign_ops[param])
            feed_dict[self._cached_assign_placeholders[param]] = value.astype(dtype)
            if debug:
                print("setting value of %s" % param.name)
        tf.get_default_session().run(ops, feed_dict=feed_dict)

    def flat_to_params(self, flattened_params, **tags):
        return unflatten_tensors(flattened_params, self.get_param_shapes(**tags))

    def __getstate__(self):
        d = Serializable.__getstate__(self)
        global load_params
        if load_params:
            d["params"] = self.get_param_values()
        return d

    def __setstate__(self, d):
        Serializable.__setstate__(self, d)
        global load_params
        if load_params:
            tf.get_default_session().run(tf.initialize_variables(self.get_params()))
            self.set_param_values(d["params"])


class JointParameterized(Parameterized):
    def __init__(self, components):
        super(JointParameterized, self).__init__()
        self.components = components

    def get_params_internal(self, **tags):
        params = [param for comp in self.components for param in comp.get_params_internal(**tags)]
        # only return unique parameters
        return sorted(set(params), key=hash)


class Model(Parameterized):
    _log_dir = './models'

    def load_params(self, exp_name, itr, skip_params):
        print 'loading policy params...'
        filename = "{}/epochs.h5".format(exp_name)
        assignments = []

        with h5py.File(filename,'r') as hf:
            if itr >= 0:
                prefix = self._prefix(itr)
            else:
                prefix = hf.keys()[itr] + "/"

            for param in self.get_params():
                path = prefix + param.name
                if param.name in skip_params:
                    continue

                if path in hf:
                    assignments.append(
                        param.assign(hf[path][...])
                        )
                else:
                    halt = True

        sess = tf.get_default_session()
        sess.run(assignments)
        print 'done.'

    def set_log_dir(self, log_dir):
		self.log_dir = log_dir

    def set_load_dir(self, load_dir):
		self.load_dir = load_dir

    @staticmethod
    def _prefix(x):
		return 'iter{:05}/'.format(x)


    def save_params(self, itr, overwrite= False):
        print 'saving model...'
        if not hasattr(self, 'log_dir'):
            log_dir = Model._log_dir
        else:
            log_dir = self.log_dir
        filename = log_dir + "/" + self.save_name + '.h5'
        sess = tf.get_default_session()

        key = self._prefix(itr)
        with h5py.File(filename, 'a') as hf:
            if key in hf:
                dset = hf[key]
            else:
                dset = hf.create_group(key)

            vs = self.get_params()
            vals = sess.run(vs)

            for v, val in zip(vs, vals):
                dset[v.name] = val
        print 'done.'

    def save_extra_data(self, names, data, itr=None):
		print 'saving model...'
		if not hasattr(self, 'log_dir'):
			log_dir = Model._log_dir
		else:
			log_dir = self.log_dir
		filename = log_dir + "/" + self.save_name + '.h5'
		sess = tf.get_default_session()
		assert len(names) == len(data)
		with h5py.File(filename, 'a') as hf:
			for name, d in zip(names,data):
				if itr is not None:
					name = self._prefix(itr) + name
				if type(d) == dict:
					self.recursively_save_dict_contents_to_group(
						hf, name, d
					)
				else:
					hf.create_dataset(name,data=d)
		
		print 'done.'
			
    @staticmethod	
    def recursively_save_dict_contents_to_group(h5file, path, dic):
		"""
		....
		"""
		for key, item in dic.items():
			if isinstance(item, (np.ndarray, np.int64, np.float64, np.int32, np.float32, str, bytes)):
				h5file[path + key] = item
			elif isinstance(item, dict):
				recursively_save_dict_contents_to_group(h5file, path + key + '/', item)
			else:
				raise ValueError('Cannot save %s type'%type(item))
			
    @staticmethod
    def load_dict_from_hdf5(filename):
		"""
		....
		"""
		with h5py.File(filename, 'r') as h5file:
			return recursively_load_dict_contents_from_group(h5file, '/')
	
    @staticmethod
    def recursively_load_dict_contents_from_group(h5file, path):
		"""
		....
		"""
		ans = {}
		for key, item in h5file[path].items():
			if isinstance(item, h5py._hl.dataset.Dataset):
				ans[key] = item.value
			elif isinstance(item, h5py._hl.group.Group):
				ans[key] = recursively_load_dict_contents_from_group(h5file, path + key + '/')
		return ans
