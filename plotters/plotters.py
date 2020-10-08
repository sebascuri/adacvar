"""Plotters for cvar experiments."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from .utilities import set_figure_params, add_text, save_or_show, set_lims
plt.switch_backend('agg')


__author__ = 'Sebastian Curi'
__all__ = ['plot_logs', 'plot_bars', 'plot_tradeoff']


def plot_logs(log, x_label=None, y_label=None, title=None, legend_loc=None,
              x_lim=None, y_lim=None, file_name=None):
    """Plot the log of an experiment.

    Parameters
    ----------
    log: dict
        dictionaries with runs.
    x_label: string, optional
        Axes x-label.
    y_label: string, optional
        Axes y-label.
    title: string, optional
        Axes plot title.
    legend_loc: string, optional
        Location of legend. If None, then no legend is plot.
    x_lim: ndarray, optional
        Limits of x-axis.
    y_lim: ndarray, optional
        Limits of y-axis.
    file_name: string, optional
        Name of file where to save the plot. If None, then show.

    """
    set_figure_params(fontsize=16)
    fig, ax = plt.subplots()
    fig.set_size_inches(np.array([4.5, 2.2]))

    for key, value in log.items():
        val = np.array(value)
        mean = np.mean(val, axis=0).ravel()
        std = np.std(val, axis=0).ravel()

        x = np.arange(len(mean))
        if key == 'adaptive':
            label = r"$\bf{adaptive}$"
        else:
            label = key

        ax.plot(x, mean, label=label)
        ax.fill_between(x, mean - std, mean + std, alpha=0.5)

    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

    add_text(ax, x_label, y_label, title, legend_loc)
    set_lims(ax, x_lim, y_lim)
    fig.tight_layout(pad=0.2)
    save_or_show(fig, file_name)
    plt.close()


def plot_bars(ax, x_labels, sampler_values, title_str=None, plot_legend=True,
              normalize=True):
    """Plot bar graph for the different values.

    Parameters
    ----------
    ax: axes
        axes where to plot the bar graph.
    x_labels: list of string
        list of labels for the x-axis.
    sampler_values: dict
        dictionary with values as {x-label: value}.
    title_str: str, optional
        title for the plot.
    plot_legend: bool
        flag that indicates if the plot legend should be shown.

    """
    x = np.arange(len(x_labels))
    values = []
    for x_label in x_labels:
        if normalize:
            max_value = np.max([a[0] for a in sampler_values[x_label].values()])
            for algorithm in sampler_values[x_label].keys():
                sampler_values[x_label][algorithm] /= max_value
        values.append(sampler_values[x_label])

    width = 0.35
    for i_s, sampler in enumerate(values[0].keys()):
        if sampler == 'adaptive':
            label = r"$\bf{adaptive}$"
        else:
            label = sampler

        ax.bar(x + (i_s - 2) * width / 2 + width / 4,
               list(a[sampler][0] for a in values),
               width / 2,
               yerr=list(a[sampler][1] for a in values),
               label=label)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    # ax.set_ylabel(y_label)
    x_labels = [label.replace('-regression', '') for label in x_labels]
    x_labels = [label[:4] for label in x_labels]

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=0)

    if title_str is not None:
        ax.set_title(title_str)
    if plot_legend:
        ax.legend(loc='best', frameon=False, ncol=4)


def plot_tradeoff(ax, cvar, accuracy, title_str=None, plot_legend=True):
    """Plot trade-off between accuracy and CVaR."""

    cvar_ = {}
    accuracy_ = {}
    for dataset in cvar:
        for algorithm in cvar[dataset]:
            if algorithm not in cvar_:
                cvar_[algorithm] = []
                accuracy_[algorithm] = []

            cvar_[algorithm].append(cvar[dataset][algorithm][0])
            accuracy_[algorithm].append(accuracy[dataset][algorithm][0])

    for algorithm in cvar_:
        ax.plot(accuracy_[algorithm], cvar_[algorithm], '*', label=algorithm)

    if title_str is not None:
        ax.set_title(title_str)
    if plot_legend:
        ax.legend(loc='best')
    else:
        ax.set_xlabel('Accuracy')
    ax.set_ylabel('CVaR')
