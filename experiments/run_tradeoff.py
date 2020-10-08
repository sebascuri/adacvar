"""Python Script Template."""
from itertools import product
import os
import pickle
from collections import defaultdict

SAVE_DIR = '/local/scuri/k_max_loss/experiments'
DIRECTORY = '/local/scuri/experiments/runs/trade-off/'
SHIFTS = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
NETWORKS = ['vgg16_bn', 'resnet18']
ALPHA = 0.1
ALGORITHMS = ['adacvar', 'mean', 'trunc_cvar', 'soft_cvar']

results = {}

for shift, network, alg in product(SHIFTS, NETWORKS, ALGORITHMS):
    d = DIRECTORY + '{}/{}/{}/{}/'.format(shift, network, ALPHA, alg)

    if shift not in results:
        results[shift] = {}
    if network not in results[shift]:
        results[shift][network] = defaultdict(list)

    for file in filter(lambda x: 'obj' in x, os.listdir(d)):
        with open(d + file, 'rb') as f:
            aux = pickle.load(f)
        print(d, file)
        results[shift][network][alg].append(aux)

with open(SAVE_DIR + '/trade-off.obj', 'wb') as file:
    pickle.dump(results, file)
