"""Process data-shift runs.

Read all experiments.
Filter experiments with largest average train accuracy.
Save data frames.

"""
import pandas as pd
import os
from itertools import product
import pickle
ALGORITHMS = ['adacvar', 'mean', 'trunc_cvar', 'soft_cvar']

df = pd.DataFrame([], columns=['dataset', 'algorithm', 'name', 'experiment_id', 'seed',
                               'train_loss', 'train_accuracy', 'train_cvar',
                               'loss', 'accuracy'])

base_dir = 'runs'
datasets = os.listdir(base_dir)
alpha = 0.1
name_idx = {'upsample': -15, '': -6}
filter_key = 'train_cvar'


def filter_func(g):
    """Filter data frame g with respect to the minimum of `filter_key`."""
    return g[g[filter_key] == g[filter_key].min()]
# filter_func = lambda g:


for data_shift, resample in product(['both', 'train', 'test', 'double'],
                                    ['', 'upsample']):
    experiment_id = 0

    for algorithm in ALGORITHMS:
        for dataset in sorted(datasets):
            log_dir = f"{base_dir}/{dataset}/0.1/linear/{alpha}/{algorithm}"
            name = ''
            seed = 0

            if not os.path.isdir(log_dir):
                continue
            for file in sorted(filter(lambda x: '.obj' in x, os.listdir(log_dir))):
                with open(log_dir + '/' + file, 'rb') as f:
                    results = pickle.load(f)

                if resample == 'upsample' and 'upsample' not in file:
                    continue
                elif resample == '' and ('upsample' in file or 'downsample' in file):
                    continue

                if data_shift not in file:
                    continue
                if name != file[:name_idx[resample]]:
                    name = file[:name_idx[resample]]
                    experiment_id += 1
                    seed = 0
                else:
                    seed += 1

                local_df = pd.DataFrame([
                    (dataset, algorithm, name, experiment_id, seed,
                     results['train']['loss'][-1][-1],
                     results['train']['accuracy'][-1][-1],
                     results['train']['cvar'][-1][-1],
                     results['test']['loss'][-1][-1],
                     results['test']['accuracy'][-1][-1])],
                    columns=['dataset', 'algorithm', 'name', 'experiment_id', 'seed',
                             'train_loss', 'train_accuracy', 'train_cvar',
                             'loss', 'accuracy'])
                df = df.append(local_df)

    # Select best hparam
    seed_avg_df = df.groupby(
        ['dataset', 'algorithm', 'experiment_id', 'seed']
    ).max().mean(
        level=[0, 1, 2]
    )  # Average over seeds.
    grouped_df = pd.DataFrame(seed_avg_df.to_records())
    experiments_id = grouped_df.groupby(['dataset', 'algorithm']).apply(
        filter_func).experiment_id.values  # Extract IDs
    df = df[df.experiment_id.isin(experiments_id)]  # Filter by best IDs.
    df.seed = df.seed.astype(int)

    df.to_pickle(f"data_frames_cvar/{data_shift}_{resample}.pkl")

# table = pd.pivot_table(df, values=['seed', 'accuracy'],
#                        index=['dataset'],
#                        columns=['algorithm'],
#                        aggfunc={'accuracy':
#                                     lambda x: f"{x.mean():.2f} \pm {x.std():.1f}"}
#                        )
