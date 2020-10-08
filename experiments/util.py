"""Python Script Template."""
import sys
import itertools
import pickle
from collections import defaultdict
from itertools import product
import os
import numpy as np
from adacvar.util.io import HTMLWriter
import matplotlib.pyplot as plt
from plotters import plot_logs, plot_bars
from plotters.utilities import save_or_show, set_figure_params
from runner import init_runner
import yaml
from scipy.stats import ttest_rel  # , ttest_ind

SAVE_DIR = os.getcwd() + '/experiments/'
ALGORITHMS = ['adacvar', 'mean', 'trunc_cvar', 'soft_cvar']


def make_commands(base_args, fixed_hyper_args, common_hyper_args, algorithm_hyper_args):
    """Generate command to run.

    Parameters
    ----------
    base_args: dict
        Base arguments to execute.
    fixed_hyper_args: dict
        Fixed hyperparameters to execute.
    common_hyper_args: dict
        Iterable hyperparameters to execute in different runs.
    algorithm_hyper_args
        Algorithm dependent hyperparameters to execute.

    Returns
    -------
    commands: list
        List with commands to execute.

    """
    interpreter_script = sys.executable
    base_cmd = interpreter_script + ' ' + 'adacvar/run.py'
    commands = []

    if common_hyper_args is None:
        common_hyper_args = dict()

    common_hyper_args = common_hyper_args.copy()
    if algorithm_hyper_args is not None:
        common_hyper_args.update(algorithm_hyper_args)

    hyper_args_list = list(dict(zip(common_hyper_args, x))
                           for x in itertools.product(*common_hyper_args.values()))

    for hyper_args in hyper_args_list:
        cmd = base_cmd
        for dict_ in [base_args, fixed_hyper_args, hyper_args]:
            for key, value in dict_.items():
                if isinstance(value, bool) and value:
                    cmd += f" --{key}"
                else:
                    cmd += " --%s=%s" % (str(key), str(value))
        commands.append(cmd)

    return commands


def run_multiple(task, algorithms, alphas, shifts, seeds, params, datasets=None):
    """Run multiple batches."""
    with open('experiments/{}.yaml'.format(task), 'r') as file:
        configs = yaml.load(file, Loader=yaml.SafeLoader)

    if datasets is None:
        datasets = configs['dataset']

    for dataset in datasets:
        cmd_list = []
        for algorithm, alpha, shift, seed in itertools.product(algorithms, alphas,
                                                               shifts, seeds):
            base_args = {'dataset': dataset, 'algorithm': algorithm, 'shift': shift,
                         'alpha': alpha, 'seed': seed,
                         'num-threads': params['num_threads']}

            new_commands = make_commands(
                base_args,
                configs['fixed_hyper_params'],
                configs['common_hyper_params'],
                configs[algorithm + '_hyper_params'])
            cmd_list += new_commands

        runner = init_runner(dataset, **params)
        runner.run_batch(cmd_list)


def get_best_log(log_dir, algorithms):
    """Get the best log in LOG_DIR."""
    log = {}
    best_hyper_params = {}
    for algorithm in algorithms:
        aux = defaultdict(list)
        for file in filter(lambda x: 'obj' in x, os.listdir(log_dir + algorithm)):
            with open(log_dir + algorithm + '/' + file, 'rb') as f:
                results = pickle.load(f)

            if np.isnan(results['validation']['cvar'][-1][-1]):
                continue

            name = file[:-6]
            seed = file[-5]

            # if algorithm == 'cvar':
            #     aux[name].append((-results['validation']['cvar'][-1][-1], results))
            # else:
            if 'accuracy' in results['test'].keys:
                aux[name].append((-results['validation']['accuracy'][-1][-1], results))
            else:
                aux[name].append((results['validation']['cvar'][-1][-1], results))
            # else:
            #     aux[name].append((results['test']['cvar'][-1][-1], results))

        best_name = None
        best_val = float('Inf')

        for name, values in aux.items():

            # print(algorithm, name, [v[0] for v in values])
            val = np.mean([v[0] for v in values])
            # print(algorithm, name, np.mean(val))
            if val < best_val:
                best_val = np.mean(val)
                best_name = name

        # print(best_name, algorithm, best_val)
        log[algorithm] = {'train': defaultdict(list), 'test': defaultdict(list),
                          'validation': defaultdict(list)}
        for seed, result in aux[best_name]:
            for key in result['test'].keys:
                log[algorithm]['test'][key].append(result['test'][key][-1][-1])
            for key in result['train'].keys:
                log[algorithm]['train'][key].append(result['train'][key][-1])
            for key in result['validation'].keys:
                log[algorithm]['validation'][key].append(result['validation'][key][-1])
        best_hyper_params[algorithm] = best_name
        print(best_val)
    return log, best_hyper_params


def writeHTML(directory, dataset, log):
    """Write log from dataset into directory in HTML format."""
    writer = HTMLWriter(directory + '/results',
                        headers={'title': 'Log of {}'.format(dataset)})
    algorithms = list(log.keys())
    header = ['metric'] + algorithms
    table = []
    # Make test table.
    for key in log[algorithms[0]]['test'].keys():
        if key == 'confusion_matrix':
            continue
        row = [key]
        for algorithm in sorted(algorithms):
            value = log[algorithm]['test'][key]
            mean = np.round(np.mean(value, axis=0), 2).ravel()[0]
            std = np.round(np.std(value, axis=0), 3).ravel()[0]
            row.append('{} ({})'.format(mean, std))

        table.append(row)
    writer.add_table(header, table, 'Test Metrics')

    header = ['metric'] + algorithms
    table = []
    # Make test table.
    for key in log[algorithms[0]]['train'].keys():
        if key == 'confusion_matrix':
            continue
        row = [key]
        for algorithm in sorted(algorithms):
            for i in range(len(log[algorithm]['train'][key])):
                aux = 0
                while len(log[algorithm]['train'][key][i]) == 1:
                    print(aux, i, algorithm, key)
                    log[algorithm]['train'][key][i] = log[algorithm]['train'][key][aux]
                    aux += 1
            try:
                value = np.array(log[algorithm]['train'][key])[:, -1]
            except IndexError:
                value = np.array(log[algorithm]['train'][key])
            mean = np.round(np.mean(value, axis=0), 2).ravel()[0]
            std = np.round(np.std(value, axis=0), 3).ravel()[0]
            row.append('{} ({})'.format(mean, std))

        table.append(row)
    writer.add_table(header, table, 'Train Metrics')

    # Make train Plots
    legend_loc = defaultdict(lambda: None)
    legend_loc['loss'] = 'best'
    y_lim = defaultdict(lambda: [0, 1.0])
    y_lim['cvar'] = [1.2, 2.6]
    y_lim['var'] = [0.2, 2.6]
    y_lim['loss'] = [0.2, 2.6]
    title = {'f1': 'F1-Score', 'var': 'VaR', 'cvar': 'CVaR'}

    for key in log[algorithms[0]]['train'].keys():
        if key == 'confusion_matrix':
            continue
        file_name = '{}.png'.format(key)
        plot_log = {a: log[a]['train'][key] for a in sorted(algorithms)}

        plot_logs(plot_log, title=title.get(key, key.capitalize()),
                  x_label='Number Epochs', y_lim=y_lim[key],
                  legend_loc=legend_loc[key], file_name=directory + '/' + file_name)

        writer.add_image(file_name, '\"width:49%\"')

    del writer


def post_process(task, algorithms, alphas, shifts, datasets=None, networks=None):
    """Post process a task."""
    if 'SCRATCH' not in os.environ:
        base_dir = os.getcwd()
    else:
        base_dir = os.environ['SCRATCH']

    run_dir = base_dir + '/experiments/runs/'
    if datasets is None:
        with open('experiments/{}.yaml'.format(task), 'r') as file:
            configs = yaml.load(file, Loader=yaml.SafeLoader)
        datasets = configs['dataset']
    if networks is None:
        networks = [None]

    logs = {}
    best_hyper_params = {}

    for dataset, alpha, shift, network in product(datasets, alphas, shifts, networks):
        key = '{}/{}/'.format(dataset, shift)
        if network:
            key += '{}/'.format(network)

        log_dir = run_dir + key + '{}/'.format(alpha)
        if key not in logs:
            logs[key] = {}
            best_hyper_params[key] = {}

        log, hyper_param = get_best_log(log_dir, algorithms)
        logs[key][alpha] = log
        best_hyper_params[key][alpha] = hyper_param
        print(key, alpha, hyper_param)
        # for algorithm in algorithms:
        #     print(algorithm)
        #     print(best_hyper_params[key][alpha][algorithm])
        #     print('cvar:',
        #           [a.pop() for a in logs[key][alpha][algorithm]['train']['cvar']])
        #     print('loss:', logs[key][alpha][algorithm]['test']['loss'])
        #     if 'accuracy' in logs[key][alpha][algorithm]['test']:
        #         print('acc:', logs[key][alpha][algorithm]['test']['accuracy'])

    with open(SAVE_DIR + '{}.obj'.format(task), 'wb') as file:
        pickle.dump((logs, best_hyper_params), file)


def plot_task(task, algorithms, alphas, shifts, plots, datasets=None):
    """Plot the given task."""
    with open('experiments/{}.obj'.format(task), 'rb') as file:
        logs = pickle.load(file)

    if datasets is None:
        with open('experiments/{}.yaml'.format(task), 'r') as file:
            configs = yaml.load(file, Loader=yaml.SafeLoader)
        datasets = configs['dataset']

    for alpha, shift in product(alphas, shifts):

        results = defaultdict(lambda: defaultdict(dict))

        for dataset in datasets:
            key = '{}/{}/'.format(dataset, shift)
            html_dir = SAVE_DIR + 'figures/' + key + str(alpha)
            log = logs[0][key][alpha]
            writeHTML(html_dir, key, logs[0][key][alpha])

            for plot in plots:
                for algorithm in algorithms:
                    aux = np.array(log[algorithm][plot[0]][plot[1]])
                    if aux.ndim > 1:
                        aux = aux[:, -1]
                    results[plot][dataset][algorithm] = (aux.mean(), aux.std(), aux)

        for plot in plots:
            set_figure_params(fontsize=20)
            fig, ax = plt.subplots()
            # fig.set_size_inches(np.array([6.75, 2.2]))

            plot_bars(ax, datasets, results[plot],
                      title_str='{} {}'.format(plot[0].capitalize(),
                                               plot[1].capitalize()),
                      plot_legend=plot[1] == 'loss',
                      normalize=False)

            fig.tight_layout(pad=0.2)
            file_dir = SAVE_DIR + 'figures/{}/{}/{}/'.format(task, shift, alpha)
            if not os.path.exists(file_dir):
                os.makedirs(file_dir)
            save_or_show(fig, file_dir + '{}_{}.png'.format(*plot))

            if (plot[0] == 'train' and plot[1] == 'cvar') or \
                    (plot[0] == 'test' and plot[1] == 'loss'):
                print(plot)
                for dataset in datasets:
                    # print(dataset)
                    aux = results[plot][dataset]

                    best_algs = [k for k, _ in
                                 sorted(aux.items(), key=lambda item: item[1][0])]
                    for i in range(1, 3):
                        alg1 = best_algs[0]
                        alg2 = best_algs[i]
                        sample1 = aux[alg1][-1]
                        sample2 = aux[alg2][-1]
                        print(dataset, alg1, alg2, aux[alg1][-1], aux[alg2][-1])
                        while len(sample1) < len(sample2):
                            sample2 = sample2[:-1]

                        while len(sample1) > len(sample2):
                            sample1 = sample1[:-1]

                        test = ttest_rel(sample1, sample2)
                        if test.pvalue < 0.05:
                            break
                    best_algs = best_algs[:i]

                    looser_format = '{:.2f} $\\pm$ {:.2f}'
                    winner_format = '\\textbf{{{{{}}}}}'.format(looser_format)
                    string = ''
                    for algorithm in ALGORITHMS:
                        if algorithm in best_algs:
                            format_ = winner_format
                        else:
                            format_ = looser_format

                        string += format_.format(aux[algorithm][0], aux[algorithm][1])
                        string += ' & '

                    # print(dataset)
                    print('& {} & '.format(dataset.capitalize()) + string[:-2] + '\\\\')


def print_task(task, algorithms, alphas, shifts, plots, datasets=None, networks=None,
               p_value=0.05):
    """Print the given task."""
    with open('experiments/{}.obj'.format(task), 'rb') as file:
        logs = pickle.load(file)

    if datasets is None:
        with open('experiments/{}.yaml'.format(task), 'r') as file:
            configs = yaml.load(file, Loader=yaml.SafeLoader)
        datasets = configs['dataset']
    if networks is None:
        networks = [None]

    totals = {algorithm: 0 for algorithm in algorithms}
    for shift, alpha, network in product(shifts, alphas, networks):
        # print(shift, alpha)
        results = defaultdict(lambda: defaultdict(dict))

        for dataset in datasets:
            key = '{}/{}/'.format(dataset, shift)
            if network:
                key += '{}/'.format(network)
            log = logs[0][key][alpha]

            for plot in plots:
                for algorithm in algorithms:
                    k = plot[1]
                    split = plot[0]
                    if split == 'train':
                        if not log[algorithm]['train'][k]:
                            log[algorithm]['train'][k] = [
                                [np.nan] * 51 for _ in range(5)]
                            # continue
                        length = max([len(a) for a in log[algorithm]['train'][k]])
                        for i in range(len(log[algorithm]['train'][k])):
                            while len(log[algorithm]['train'][k][i]) < length:
                                log[algorithm]['train'][k][i] = np.concatenate(
                                    (log[algorithm]['train'][k][i],
                                     log[algorithm]['train'][k][i][-1:]))

                    aux = np.array(log[algorithm][plot[0]][plot[1]])
                    aux = np.delete(aux, np.where(np.isnan(aux)))

                    if aux.ndim > 1:
                        aux = aux[:, -1]
                    results[plot][dataset][algorithm] = (aux.mean(), aux.std(), aux)

        for dataset in datasets:
            strings = []
            for plot in plots:
                # print(dataset, plot)
                aux = results[plot][dataset]

                best_algs = [k for k, _ in
                             sorted(aux.items(), key=lambda item: item[1][0])]
                if plot[1] == 'accuracy' or plot[1] == 'precision' or plot[1] == 'f1':
                    best_algs = [a for a in reversed(best_algs)]
                for i in range(1, 3):
                    sample1 = aux[best_algs[0]][-1]
                    sample2 = aux[best_algs[i]][-1]
                    while len(sample1) < len(sample2):
                        sample2 = sample2[:-1]

                    while len(sample1) > len(sample2):
                        sample1 = sample1[:-1]

                    test = ttest_rel(sample1, sample2)
                    if test.pvalue < p_value:
                        break
                best_algs = best_algs[:i]

                looser_format = '{:.2f} $\\pm$ {:.1f}'
                winner_format = '\\winner{{{{{}}}}}'.format(looser_format)
                string = []
                for algorithm in ALGORITHMS:
                    if algorithm in best_algs:
                        format_ = winner_format
                        totals[algorithm] += 1
                    else:
                        format_ = looser_format

                    string.append(format_.format(aux[algorithm][0], aux[algorithm][1]))
                strings.append(string)

            ds = '& {} & '.format(dataset.capitalize())
            if len(strings) == 1:
                print(ds + ' & '.join(*strings) + ' \\\\')
            else:
                print(ds + ' '.join(
                    [a + ' & ' + b + ' & ' for a, b in zip(*strings)])[:-2] + '\\\\')
    # print(totals)
