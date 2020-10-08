"""Utilities for plotter module."""

import matplotlib.pyplot as plt
from matplotlib.colors import ColorConverter
from matplotlib import rcParams
import numpy as np
import pickle

__author__ = 'Felix Berkenkamp and Sebastian Curi'
__all__ = ['add_text', 'set_lims', 'set_figure_params', 'hide_all_ticks',
           'hide_spines', 'set_frame_properties', 'linewidth_in_data_units',
           'adapt_figure_size_from_axes', 'emulate_color', 'save_or_show',
           'cm2inches', 'inches2cm'
           ]


def add_text(axis, x_label=None, y_label=None, title=None, legend_loc=None):
    """Edit text on axis.

    Parameters
    ----------
    axis: matplotlib axis
        Axis to modify.
    x_label: string, optional
        Axes x-label.
    y_label: string, optional
        Axes y-label.
    title: string, optional
        Axes plot title.
    legend_loc: string, optional
        Location of legend. If None, then no legend is plot.
    """
    if x_label is not None:
        axis.set_xlabel(x_label)
    if y_label is not None:
        axis.set_ylabel(y_label)
    if title is not None:
        axis.set_title(title)
    if legend_loc is not None:
        axis.legend(loc=legend_loc, frameon=False, ncol=2,
                    )


def set_lims(axis, x_lim=None, y_lim=None):
    """Set axis limits.

    Parameters
    ----------
    axis: matplotlib axis
        Axis to modify.
    x_lim: ndarray, optional
        Limits of x-axis.
    y_lim: ndarray, optional
        Limits of y-axis.

    """
    if x_lim is not None:
        axis.set_xlim(x_lim)

    if y_lim is not None:
        axis.set_ylim(y_lim)


def set_figure_params(serif=False, fontsize=9):
    """Define default values for font, fontsize and use latex.

    Parameters
    ----------
    serif: bool, optional
        Whether to use a serif or sans-serif font.
    fontsize: int, optional
        Size to use for the font size

    """
    params = {
        'font.serif': ['Times',
                       'Palatino',
                       'New Century Schoolbook',
                       'Bookman',
                       'Computer Modern Roman'] + rcParams['font.serif'],
        'font.sans-serif': ['Times',
                            'Helvetica',
                            'Avant Garde',
                            'Computer Modern Sans serif'
                            ] + rcParams['font.sans-serif'],
        'font.family': 'sans-serif',
        'text.usetex': True,
        # Make sure mathcal doesn't use the Times style
        #  'text.latex.preamble':
        # r'\DeclareMathAlphabet{\mathcal}{OMS}{cmsy}{m}{n}',

        'axes.labelsize': fontsize,
        'axes.linewidth': .75,
        'font.size': fontsize,
        'legend.fontsize': fontsize * 0.7,
        'xtick.labelsize': fontsize * 8 / 9,
        'ytick.labelsize': fontsize * 8 / 9,
        'figure.dpi': 100,
        'savefig.dpi': 600,
        'legend.numpoints': 1,
    }

    if serif:
        params['font.family'] = 'serif'

    rcParams.update(params)


def hide_all_ticks(axis):
    """Hide all ticks on the axis.

    Parameters
    ----------
    axis: matplotlib axis
        Axis to modify.

    """
    axis.tick_params(axis='both',  # changes apply to the x-axis
                     which='both',  # affect both major and minor ticks
                     bottom=False,  # ticks along the bottom edge are off
                     top=False,  # ticks along the top edge are off
                     left=False,  # No ticks left
                     right=False,  # No ticks right
                     labelbottom=False,  # No tick-label at bottom
                     labelleft=False)  # No tick-label at bottom


def hide_spines(axis, top=True, right=True):
    """Hide the top and right spine of the axis.

    Parameters
    ----------
    axis: matplotlib axis
        Axis to modify.
    top: bool, optional
        If true, hide top spine.
    right: bool, optional
        If true, hide right spine.

    """
    if top:
        axis.spines['top'].set_visible(False)
        axis.xaxis.set_ticks_position('bottom')
    if right:
        axis.spines['right'].set_visible(False)
        axis.yaxis.set_ticks_position('left')


def set_frame_properties(axis, color, lw):
    """Set color and linewidth of frame.

    Parameters
    ----------
    axis: matplotlib axis
        Axis to modify.
    color: string
        Color of frame.
    lw: int
        Line width of frame.

    """
    for spine in axis.spines.values():
        spine.set_linewidth(lw)
        spine.set_color(color)


def linewidth_in_data_units(lw, axis, reference='y'):
    """Convert a linewidth in data units to linewidth in points.

    Parameters
    ----------
    lw: float
        Linewidth in data units of the respective reference-axis.
    axis: matplotlib axis
        The axis which is used to extract the relevant transformation
        data (data limits and size must not change afterwards).
    reference: string
        The axis that is taken as a reference for the data width.
        Possible values: 'x' and 'y'. Defaults to 'y'.

    Returns
    -------
    lw: float
        Linewidth in points
    """
    fig = axis.get_figure()

    if reference == 'x':
        # width of the axis in inches
        axis_length = fig.get_figwidth() * axis.get_position().width
        value_range = np.diff(axis.get_xlim())
    elif reference == 'y':
        axis_length = fig.get_figheight() * axis.get_position().height
        value_range = np.diff(axis.get_ylim())

    # Convert axis_length from inches to points
    axis_length *= 72

    return (lw / value_range) * axis_length


def adapt_figure_size_from_axes(axes):
    """Adapt the figure sizes so that all axes are equally wide/high.

    When putting multiple figures next to each other in Latex, some
    figures will have axis labels, while others do not. As a result,
    having the same figure width for all figures looks really strange.
    This script adapts the figure sizes post-plotting, so that all the axes
    have the same width and height.

    Be sure to call plt.tight_layout() again after this operation!

    This doesn't work if you have multiple axis on one figure and want them
    all to scale proportionally, but should be an easy extension.

    Parameters
    ----------
    axes: list of matplotlib axis
        List of axes that we want to have the same size
        (need to be on different figures).

    """
    # Get parent figures
    figures = [axis.get_figure() for axis in axes]

    # get axis sizes [0, 1] and figure sizes [inches]
    axis_sizes = np.array([axis.get_position().size for axis in axes])
    figure_sizes = np.array([figure.get_size_inches() for figure in figures])

    # Compute average axis size [inches]
    avg_axis_size = np.average(axis_sizes * figure_sizes, axis=0)

    # New figure size is the average axis size plus the white space that is
    # not begin used by the axis so far (e.g., the space used by labels)
    new_figure_sizes = (1 - axis_sizes) * figure_sizes + avg_axis_size

    # Set new figure sizes
    for figure, size in zip(figures, new_figure_sizes):
        figure.set_size_inches(size)


def emulate_color(color, alpha=1, background_color=(1, 1, 1)):
    """Take an RGBA color and an RGB background, return the emulated RGB color.

    The RGBA color with transparency alpha is converted to an RGB color via
    emulation in front of the background_color.
    """
    to_rgb = ColorConverter().to_rgb
    color = to_rgb(color)
    background_color = to_rgb(background_color)
    return [(1 - alpha) * bg_col + alpha * col
            for col, bg_col in zip(color, background_color)]


def save_or_show(fig, file_name=None):
    """Decide whether to save or show a figure.

    If file name is not None, then save the figure.
    Else show the figure.

    Parameters
    ----------
    fig: matplotlib figure handle
        Figure to show or save
    file_name: str, optional
        File name of saved figure.

    """
    if file_name is None:
        fig.show()
        plt.show()
    else:
        with open(file_name[:file_name.rfind('.')] + '.pkl', 'wb') as file:
            pickle.dump(fig, file)

        fig.savefig(file_name)


def cm2inches(centimeters):
    """Convert centimeters to inches.

    Parameters
    ----------
    centimeters: ndarray or float

    Returns
    -------
    inches: ndarray or float

    """
    return centimeters / 2.54


def inches2cm(inches):
    """Convert inches to centimeters.

    Parameters
    ----------
    inches: ndarray or float

    Returns
    -------
    centimeters: ndarray or float

    """
    return inches * 2.54
