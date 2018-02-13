import numpy as np
import tensorflow as tf
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, adjusted_mutual_info_score

from rllab.config import EXPERT_PATH
from trn.config import best_epochs

from scipy.stats import mode, entropy

import argparse
import h5py

from myscripts import _create_expert_data, _create_env, _restore_model_args, _create_aux_networks

from sklearn.svm import SVC

parser = argparse.ArgumentParser()

# Logger Params
parser.add_argument('--environment',type=str,default='JTZM')
parser.add_argument('--mix_data_classes',type=str,default=1)
parser.add_argument('--max_path_length',type=int,default=50)
parser.add_argument('--z_dim',type=int,default=4)
parser.add_argument('--use_infogail',type=int,default=1)
parser.add_argument('--domain_indices',type=int,default=[1])
parser.add_argument('--model_all',type=int,default=0)
parser.add_argument('--use_valid',type=int,default=1)
args = parser.parse_args()

#exp_name0 = "../data/models/17-06-07/EXPERIMENT4-06070746-JTZM-0"
#exp_name1 = "../data/models/17-06-07/EXPERIMENT6-06091833-JTZM-9"
#exp_name1 = "../data/models/17-06-10/EXPERIMENT6-06092103-JTZM-3"

exp_name0 = "../data/models/17-06-13/CORL2-06130728-JTZM-7"
#exp_name0 = "../data/models/17-06-13/CORL2-06130728-JTZM-4"

args = _restore_model_args(exp_name0, args, exclude_keys=[
    "domain_indices",
    "use_valid",
    "mix_data_classes"])

np.random.seed(10)


class Model(object):
    def __init__(self):
        pass

    def fit(self):
        pass

    @property
    def is_recurrent(self):
        pass

    @property
    def name(self):
        pass

class Random_Model(Model):
    def __init__(self, k = 4):
        self._k = k

    def fit(self, obs, act, cls):
        pass

    def predict(self, obs, act):
        N = obs.shape[0]
        return np.random.randint(0,self._k,(N,))

    @property
    def is_recurrent(self):
        return False

    @property
    def name(self):
        return "random"

class KMeans_Model(Model):
    def __init__(self, k = 4):
        self._model = KMeans(n_clusters = k)

    def fit(self, obs, act, cls):
        X = np.column_stack([obs, act])
        print("Fitting K means ...")
        self._model.fit(X)

    def predict(self, obs, act):
        X = np.column_stack([obs, act])
        print("Predicting K means ...")
        Y_B = self._model.predict(X)
        return Y_B

    @property
    def z_dim(self):
        return self._z_dim

    @property
    def is_recurrent(self):
        return False

    @property
    def name(self):
        return "k-means"

class SVM_Model(Model):
    def __init__(self, k = 4):
        self._model = SVC()

    def fit(self, obs, act, cls):
        X = np.column_stack([obs, act])
        print("Fitting SVM ...")
        self._model.fit(X, cls)

    def predict(self, obs, act):
        X = np.column_stack([obs, act])
        print("Predicting SVM ...")
        Y_B = self._model.predict(X)
        return Y_B

    @property
    def z_dim(self):
        return self._z_dim

    @property
    def is_recurrent(self):
        return False

    @property
    def name(self):
        return "SVM"

class INFO_Model(Model):
    def __init__(self, info_model, exp_name):
        self._model = info_model
        epoch = best_epochs[exp_name]
        self.exp_name = exp_name
        self._model.load_params(exp_name, epoch, [])

    #def predict(self, XZ, A, Y=None, **kwargs):
    def predict(self, obs, act):
        N, _ = obs.shape
        pad = np.zeros((N,self._model.z_dim))
        X = np.column_stack([obs,pad])
        Y_B, _ = self._model.predict(X, act)
        return Y_B

    def fit(self, obs, act, cls):
        pass

    @property
    def is_recurrent(self):
        return False

    @property
    def name(self):
        #return "vae + {} ({})".format(mname, self.train_mix)
        return "info ({})".format(self.exp_name)

class VAE_Model(Model):
    def __init__(self, k = 4, train_mix = 0, supervised = 0, keep = None):
        self._k = k
        self._preds = None
        self.train_mix = ["single","mix"][train_mix]
        self.keep = keep

        if not supervised:
            self._model = KMeans(n_clusters = k)
        else:
            self._model = SVC()
        self.supervised = supervised

    def _retrieve_encodings(self, obs, act):
        B, T, Do = obs.shape
        B, T, Da = act.shape

        name = \
            "{}/{}/{}/{}_vae_trajs.h5".format(
                    EXPERT_PATH,
                    ["juliaTrack_single","juliaTrack_mix"][args.mix_data_classes],
                    ["train","valid"][args.use_valid],
                    self.train_mix)

        with h5py.File(name,"r") as hf:

            obs_vae = hf['obs_B_T_Do'][...]
            act_vae = hf['a_B_T_Da'][...]

            z_mu = hf['zmean_B_Dz'][...]
            z_logstd = hf['zlogstd_B_Dz'][...]

            obs_vae = obs_vae[self.keep]
            act_vae = act_vae[self.keep]
            z_mu = z_mu[self.keep]
            z_logstd = z_logstd[self.keep]

        assert all((obs_vae == obs).flatten())
        assert all((act_vae == act).flatten())

        return z_mu

    def predict(self, obs, act):
        z_mu = self._retrieve_encodings(obs, act)
        Y_B = self._model.predict(z_mu)
        return Y_B

    def fit(self, obs, act, cls):
        # keys: [u'a_B_T_Da', u'cls_B', u'len_B', u'obs_B_T_Do', u'zlogstd_B_T_Dz', u'zmean_B_T_Dz']
        z_mu = self._retrieve_encodings(obs, act)

        # note: JUST using the mean
        if self.supervised:
            self._model.fit(z_mu, cls)
        else:
            self._model.fit(z_mu)

    @property
    def z_dim(self):
        return self._z_dim

    @property
    def is_recurrent(self):
        return True

    @property
    def name(self):
        mname = ["k-means", "SVM"][self.supervised]
        return "vae + {} ({})".format(mname, self.train_mix)

if __name__ == "__main__":
    env = _create_env(args)
    _, reward, info_model, env = _create_aux_networks(args, env)

    expert_data_T, expert_data_V = _create_expert_data(args)
    if not args.use_valid:
        expert_data = expert_data_T
    else:
        expert_data = expert_data_V

    obs_B_T_Do = expert_data['obs']
    act_B_T_Da = expert_data['act']
    len_B = np.minimum(expert_data['exlen_B'], args.max_path_length)
    # get labels on track dataset
    cls_B = expert_data.get('cls_B',None)
    dom_B = expert_data.get('dom_B',None)
    keep = expert_data["keep"]

    B, _, Do = obs_B_T_Do.shape
    B, _, Da = act_B_T_Da.shape

    # truncate trajectories
    T = args.max_path_length
    obs_B_T_Do = obs_B_T_Do[:,:T,:]
    # have actions been normalized ?
    act_B_T_Da = act_B_T_Da[:,:T,:]

    assert obs_B_T_Do.ndim == 3
    assert act_B_T_Da.ndim == 3

    with tf.Session() as sess:
        models = [Random_Model(k = args.z_dim),
                 KMeans_Model(k = 4),
                 VAE_Model(k = args.z_dim, train_mix = 1, keep=keep),#, train_mix = 1),
                 INFO_Model(info_model, exp_name0),
                 #INFO_Model(info_model, exp_name1),
                 SVM_Model(k = 4),
                 VAE_Model(k = args.z_dim, train_mix = 1, supervised = 1,
                     keep=keep)]

        for model in models:
            if not model.is_recurrent:
                cls_B_T = np.repeat(cls_B[...,None], T, axis=1)[...,None]
                cls_BT = np.squeeze(np.reshape(cls_B_T, (B*T, 1)))
                obs_BT_Do = np.reshape(obs_B_T_Do, (B * T, Do))
                act_BT_Da = np.reshape(act_B_T_Da, (B * T, Da))

                obs_BT_Do = obs_BT_Do[~np.isnan(obs_BT_Do).any(axis=1)]
                act_BT_Da = act_BT_Da[~np.isnan(act_BT_Da).any(axis=1)]

                model.fit(obs_BT_Do, act_BT_Da, cls_BT)
                clus_BT = model.predict(obs_BT_Do, act_BT_Da)
                clus_B_T = np.reshape(clus_BT, (B,T))
                clus_B, _ = np.squeeze(mode(clus_B_T,axis=1))

            if model.is_recurrent:
                model.fit(obs_B_T_Do, act_B_T_Da, cls_B)
                clus_B = model.predict(obs_B_T_Do, act_B_T_Da)

            nmi_score = normalized_mutual_info_score(clus_B, cls_B)
            cls_ami_score = adjusted_mutual_info_score(clus_B, cls_B)
            dom_ami_score = adjusted_mutual_info_score(clus_B, dom_B)
            print("model: {} ; z-ami: {} ; d-ami: {}".format(model.name,
                cls_ami_score, dom_ami_score))

