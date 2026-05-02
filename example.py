from KSKM_CLR_SuperNodes import KSKM
import numpy as np
import pickle

data_dir = 'data/'
with open(data_dir + f'ml_super_nodes.pkl', 'rb') as f:
        ml_supernodes = pickle.load(f)
with open(data_dir + f'cl_super_nodes.pkl', 'rb') as f:
        cl_supernodes = pickle.load(f)


X = np.load(data_dir + f'X.npy')
Y = np.load(data_dir + f'Y.npy')
k = 300
random_state = 42

membership = None

membership_final, membership = KSKM(random_state, X, Y, ml_supernodes = ml_supernodes, cl_supernodes = cl_supernodes, steps_mutation = 5000, k = k, steps_back_to_best = 10, steps_no_improvement = 100, verbose = True, time_limit = 6 * 3600, membership = membership, reposition_frequency = 5, random_centroid_scale = 5, weight_supernodes = True)


