# Copyright 2017 Marco Galardini and John Lees

'''Utilities'''

import os
import contextlib
from decimal import Decimal

from matplotlib import pyplot as plt


# thanks to Laurent LAPORTE on SO
@contextlib.contextmanager
def set_env(**environ):
    """
    Temporarily set the process environment variables.

    >>> with set_env(PLUGINS_DIR=u'test/plugins'):
    ...   "PLUGINS_DIR" in os.environ
    True

    >>> "PLUGINS_DIR" in os.environ
    False
    """
    old_environ = dict(os.environ)
    os.environ.update(environ)
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(old_environ)


# avoid numpy taking up more than one thread
with set_env(MKL_NUM_THREADS='1',
             NUMEXPR_NUM_THREADS='1',
             OMP_NUM_THREADS='1'):
    import numpy as np


def format_output(item, lineage_dict=None, model='seer', print_samples=False):
    """Format results for a variant for stdout printing

    Args:
        item (pyseer.classes.Seer or pyseer.classes.LMM)
            Variant results container
        lineage_dict (list)
            Lineage labels
        model (str)
            The model used
        print_samples (bool)
            Whether to add the samples list to the putput

    Returns:
        out (str)
            Tab-delimited string to be printed
    """
    out = '%s' % item.kmer

    if model == "enet" or model == "rf":
        out += '\t' + '\t'.join(['%.2E' % Decimal(x)
                                 if np.isfinite(x)
                                 else ''
                                 for x in (item.af,
                                           item.prep,
                                           item.pvalue,
                                           item.kbeta)])
    else:
        out += '\t' + '\t'.join(['%.2E' % Decimal(x)
                                 if np.isfinite(x)
                                 else ''
                                 for x in (item.af,
                                           item.prep,
                                           item.pvalue,
                                           item.kbeta,
                                           item.bse)])
        if model == 'lmm':
            if np.isfinite(item.frac_h2):
                frac_h2 = '%.2E' % Decimal(item.frac_h2)
            else:
                frac_h2 = ''
            out += '\t' + frac_h2
        else:
            if np.isfinite(item.intercept):
                intercept = '%.2E' % Decimal(item.intercept)
            else:
                intercept = ''
            out += '\t' + intercept + '\t'
            # No distances may not have further betas
            if not np.all(np.equal(item.betas, None)):
                out += '\t'.join(['%.2E' % Decimal(x)
                                 if np.isfinite(x)
                                 else ''
                                 for x in item.betas])

    if lineage_dict is not None:
        if item.max_lineage is not None and np.isfinite(item.max_lineage):
            out += '\t' + lineage_dict[item.max_lineage]
        else:
            out += '\tNA'
    if print_samples:
        out += '\t' + '\t'.join((','.join(item.kstrains),
                                 ','.join(item.nkstrains)))
    out += '\t%s' % ','.join(item.notes)

    if model == "enet":
        out += f'\t{item.beta_idx}'

    return out

def save_fig(fig, save_dir, name="", both_formats=False, close=True):
    os.makedirs(save_dir, exist_ok=True)
    fig.savefig(os.path.join(save_dir, f"{name}.pdf"), bbox_inches='tight', dpi=300)
    
    if both_formats:
        fig.savefig(os.path.join(save_dir, f"{name}.png"), bbox_inches='tight', dpi=300)
    if close:
        plt.close(fig)  # otherwise figure may hang around in memory


def plot(title, xlabel, ylabel, figsize=(6, 4), xscale="linear", yscale="linear", use_grid=True):
    """A super basic plotting function"""
    fig, ax = plt.subplots(figsize=figsize)
    ax.set(xscale=xscale, yscale=yscale, xlabel=xlabel, ylabel=ylabel)  # a convenient way to set lots of properties at once!
    ax.grid(use_grid)
    fig.suptitle(title)  # why not fig.set_title()? I have no idea, and this seems like a bad design choice to me...
    
    return fig, ax


def subplots(nrows, ncols, title=None, xlabel=None, ylabel=None, width=9, height=6,
             scale="auto", sharex=True, sharey=True, grid=True, max_width=12, tight_layout=True):
    """A function for making nicely formatted grids of subplots with shared labels on the x-axis and y axis"""
    assert nrows > 1 or ncols > 1, "use plot() to create a single subplot"
    
    fig_width = width * ncols
    fig_height = height * nrows
    
    if scale == "auto":
        scale = min(1.0, max_width/fig_width)  # width of figure cannot exceede max_width inches
        scale = min(scale, max_width/fig_height)  # height of figure cannot exceede max_width inches
    
    figsize=(scale * fig_width, scale * fig_height)
    fig, axs = plt.subplots(nrows, ncols, figsize=figsize, sharex=sharex, sharey=sharey)
    
    for ax in axs.flat:
        ax.grid(grid)  # maybe add grid lines
        
    if title: fig.suptitle(title)
    if xlabel: fig.supxlabel(xlabel)  # shared x label
    if ylabel: fig.supylabel(ylabel)  # shared y label
    
    fig.tight_layout()  # adjust the padding between and around subplots to neaten things up
    
    return fig, axs

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__