#!/usr/bin/python
"""
Collection of functions to do useful (neuro)scientific data plotting 
TODO 
- Smart text
- smarter text placement for r values
- automatic axis label placement
- intelligent default axis ticks
- axis within axis, unified across suubplots
- horizontal lines with axhline, for behavioral events
"""
from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
from itertools import izip
import numpy as np
import os.path as path
from pylab import Circle
import os
from collections import OrderedDict

## Colors 
colors = OrderedDict()
colors['gray']          = (0.5, 0.5, 0.5)
colors['black']         = (0., 0., 0.)
colors['blue']          = (0., 0., 1.)
colors['green']         = (0, 0.5, 0.)
colors['fuchsia']       = (1., 0., 1.)
colors['red']           = (1., 0., 0.)
colors['teal']          = (0, 0.5, 0.5)
colors['purple']        = (0.5, 0., 0.5)
colors['DarkSlateGray'] = np.array([47., 79.,  79.])/255
colors['yellow']        = (1., 1., 0.)
colors['olive']         = (0.5, 0.5, 0.)
colors['SteelBlue']     = np.array([70., 130., 180.])/255
colors['lime']          = (0., 1., 0.)
colors['aqua']          = (0., 1., 1.)
colors['navy']          = (0., 0., 0.5)
colors['orange']        = np.array([255., 69., 0])/255
colors['Maroon']        = (0.5, 0., 0.)
colors['DarkViolet']    = np.array([148., 0., 211.])/255
colors['Aquamarine']    = np.array([127., 255., 212.])/255
colors['PeachPuff']     = np.array([255., 218., 185])/255

color_names = colors.keys()
n_colors = len(color_names)

## Constants 
panel_letter_size = 12
title_size = 12
tick_label_size = 10
axis_label_size = 10
legend_size = 8

beamer_figsize = (5,3)
ppt_figsize = (9,5)

font_sizes = {
    'paper': {
        'panel_letter': 12,
        'title': 12,
        'tick_label': 10,
        'axis_label': 10, 
        'legend': 8,
    },
    'ppt': {
        'panel_letter': 12,
        'title': 12,
        'tick_label': 20,
        'axis_label': 20, 
        'legend': 8,
    }
}


def figure(figsize=None, **kwargs):
    if isinstance(figsize, str):
        if figsize == 'ppt': figsize = ppt_figsize
        elif figsize == 'beamer': figsize == beamer_figsize
    elif isinstance(figsize, tuple):
        pass
    elif figsize == None:
        pass
    else:
        raise TypeError("Unrecognized type for figsize: %s" % type(figsize))
    return plt.figure(figsize=figsize, **kwargs)

def sem_line(ax, val, sem, orientation='horiz', **kwargs):
    if orientation == 'vert' or orientation == 'v' or orientation == 'vertical':
        ax.axvline(val + sem, **kwargs)
        ax.axvline(val - sem, **kwargs)
    elif orientation == 'horiz' or orientation == 'h' or orientation == 'horizontal':
        ax.axhline(val + sem, **kwargs)
        ax.axhline(val - sem, **kwargs)
    else:
        raise Exception("unknown orientation: %s" % orientation)

def add_cax(ax, fig, left_offset=0.03):
    """ Add colorbar axis to the right of an axis 
    """
    bbox = ax.get_position()
    cax = fig.add_axes([bbox.x1+left_offset, bbox.y0, 0.03, bbox.y1-bbox.y0])
    return cax

def label(ax, pt, text):
    raise Exception("FINISH")

## Tools for manipulating matplotlib figures 
def subplots(n_vert_plots, n_horiz_plots, x=0.03, y=0.05, left_offset=0.06, 
    aspect=None, bottom_offset=0.075, return_flat=False, show_ur_lines=False,
    hold=False, ylabels=False, sep_ylabels=False, xlabels=False, 
    sep_xlabels=False, letter_offset=None):
    """
    An implementation of the subplots function

    Parameters
    ==========

    Returns
    =======
    """
    if bottom_offset == None:
        bottom_offset = left_offset
    vert_frac_per_row = (1.-bottom_offset)/n_vert_plots
    horiz_frac_per_col = (1.-left_offset)/n_horiz_plots
    subplot_width = horiz_frac_per_col - x 
    subplot_height = vert_frac_per_row - y

    axes = []
    for m in range(n_vert_plots):
        axes_row = []
        for n in range(n_horiz_plots):
            xstart = left_offset + x/2 + horiz_frac_per_col*n 
            ystart = bottom_offset + y/2 + (n_vert_plots - 1 - m)*vert_frac_per_row
            if aspect is not None:
                new_ax = plt.axes([xstart, ystart, subplot_width, subplot_height], aspect=aspect)
            else:
                new_ax = plt.axes([xstart, ystart, subplot_width, subplot_height])

            if not show_ur_lines:
                elim_ur_lines(new_ax)

            # set hold property for ax
            new_ax.hold(hold)

            axes_row.append(new_ax)
        axes.append(axes_row)

    axes = np.array(axes)
    if not letter_offset == None:
        axis_letters = [letter_axis(ax, chr(k+65), **letter_offset) for k, ax in enumerate(axes.ravel())]

    if return_flat:
        axes = axes.ravel()

    return axes

def gen_radial_axes(n_targ=8, x_max=0.3, y_max=0.15, plot_start_rad=0.2,
        offset=0.44):
    """
    Creates N bar graphs on the same canvas with each subplot arranged
    symmetrically around the center. 

    Parameters
    ----------
    data : 2d array
        data.shape[0] defines the number of plots, data.shape[1] defines
        the number of variables
    labels : tuple of strings
        1 label for each of the variables in data.shape[1]
    err : 2D array
        Error bars for the variables in data. Same dimensions/conventions
    """
    x_max, y_max = (0.075, 0.15) # set subplot size as fraction of each dim
    plot_start_rad = 0.3

    axes = [None]*n_targ
    for k in range(0, n_targ):
        # create axes at appropriate radial location
        x_min, y_min = plot_start_rad*cos(2*pi/n_targ*k)+0.44, plot_start_rad*sin(2*pi/n_targ*k)+0.44
        axes[k] = plt.axes( [x_min, y_min, x_max, y_max] )
    return axes        


def clear_ax_labels(ax_ls, axes='xy'):
    if isinstance(ax_ls, plt.Axes):
        ax_ls = [ax_ls]
    for ax in ax_ls:
        if axes == 'xy' or axes == 'x':
            ax.set_xticks([])
            ax.set_xticklabels([])

        if axes == 'xy' or axes == 'y':
            ax.set_yticks([])
            ax.set_yticklabels([])
    
def line_across_axes(axes):
    """ Vertical line spanning y axes """
    raise Exception("FINISH")

def shade(ax, region=[None, None, None, None]):
    """ Shade a rectangular region specified 
    """
    if region == [None, None, None, None]:
        return 
    else:
        raise Exception("FINISH")
    p = plt.axhspan(0.25, 0.75, facecolor='0.5', alpha=0.5)

def error_line(ax, x_data, y_data, error_region, **kwargs):
    color = kwargs.pop('color', 'blue')
    ax.plot(x_data, y_data, color=color)
    ax.fill_between(x_data, y_data - error_region, y_data + error_region, 
        color=color, alpha=kwargs.pop('alpha', 0.15))

def contour_2D(ax, fn, N=100, **kwargs):
    """ 2d contour plot
    fn should take 2 arguments
    """
    default_arg_range = [0, 1]
    x_range = kwargs.pop('x_range', default_arg_range)
    y_range = kwargs.pop('y_range', default_arg_range)
    theta_0 = np.linspace(x_range[0], x_range[1], N)
    theta_1 = np.linspace(y_range[0], x_range[1], N)
    J = np.zeros((N,N))
    
    scale = 1./(2*pi*sqrt(np.linalg.det(P_pos)))
    for i in range(1,N):
        for j in range(1,N):
            J[i, j] = fn( theta_0[i], theta_1[j] )

    ax.contour(theta_0, theta_1, J.T) 

def elim_lines(ax, lines=['top', 'right']):
    for l in lines:
        if l == 'top':
            ax.spines['top'].set_visible(False)
            ax.xaxis.set_ticks_position('bottom')
        elif l == 'right':
            ax.spines['right'].set_visible(False)
            ax.yaxis.set_ticks_position('left')
        elif l == 'left':
            ax.spines['left'].set_visible(False)
        elif l == 'bottom':
            ax.spines['bottom'].set_visible(False)

def set_ticklabel_size(ax, xsize=tick_label_size, ysize=None):
    if ysize == None:
        ysize = xsize
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(xsize)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(ysize)

def init_ax(ax=None, aspect=None):
    """ Helper function to handle optional 'ax' arguments for other 
    functions
    """
    if ax == None:
        fig = plt.figure()
        if aspect is not None: 
            ax = plt.subplot(111, aspect=aspect)
        else:
            ax = plt.subplot(111)
    return ax
def gen_irregular_axes(axes_prop, spacing=0.05, offset=0.05, axis='vert', share=False):
    if not axis in ['horiz', 'vert']:
        raise Exception("'axis' must be either 'horiz' or 'vert'")

    axes_prop = np.array(axes_prop).ravel()
    axes_prop = axes_prop[::-1]
    axes_prop /= sum(axes_prop) 
    axes_prop *= (1-offset*2)
    axes_starts = np.r_[0, np.cumsum(axes_prop)]

    if axis=='vert':
        stuff = [ (s, e-s-spacing) for s, e in izip(axes_starts[:-1], axes_starts[1:])]
        rectangles = [ [spacing, offset+ystart, 1-spacing*2, height] for ystart, height in stuff]
    else:
        stuff = [ (s, e-s-spacing) for s, e in izip(axes_starts[:-1], axes_starts[1:])]
        rectangles = [ [ystart, spacing, height, 1-spacing] for ystart, height in stuff]

    axes = [None]*len(rectangles)
    for k,r in enumerate(rectangles):
        if share and k > 0 and axis == 'vert':
            axes[k] = plt.axes(r, sharex=axes[0])
        elif share and k > 0 and axis=='horiz':
            axes[k] = plt.axes(r, sharey=axes[0])
        else:
            axes[k] = plt.axes(r)

        if k > 0 and share and axis=='vert':
            #axes[k].set_xticks([])
            #axes[k].set_xticklabels([])
            for label in axes[k].get_xticklabels():
                label.set_visible(False)


    return axes            

def gradient_line_plot(x_coords, y_coords, ax=None, start_color=(0,1,0), end_color=(1,0,0)):
    assert len(x_coords) == len(y_coords)
    n_segments = len(x_coords)-1
    start_color = np.array(start_color)
    end_color = np.array(end_color)
    colors = [ (k/n_segments, 1-k/n_segments, 0) for k in np.arange(n_segments, dtype=np.float64)]
    if ax == None:
        ax = plt.subplot(1,1,1)
    for k in range(n_segments):
        ax.plot(x_coords[k:k+2], y_coords[k:k+2], color=colors[k])

def color_gradient(n_pts, start_color=(0,1,0), end_color=(1,0,0)):
    start_color = np.array(start_color)
    end_color = np.array(end_color)
    colors = [ (k/n_pts, 1-k/n_pts, 0) for k in np.arange(n_pts, dtype=np.float64)]
    return colors

def _plot_one_bar(ax, data, ctr, fn=np.mean, width=0.05, c='blue', s=60, m='o', 
    label=None, show_data=True, data_to_show='pts'):
    """ Generate a single bar plot
    """ 
    n_data_pts = len(data)
    ax.bar( ctr-width/2, fn(data), width=width, color='none', edgecolor=c, 
        linewidth=3)
    if show_data:
        if data_to_show == 'pts':
            ax.scatter(np.ones(n_data_pts)*ctr, data, c=c, s=s, marker=m, 
                label=label)
        elif data_to_show == 'std':
            std = np.std(data)
            ax.plot([ctr, ctr], [fn(data)-std, fn(data)+std], 
                color=c, marker=m, label=label, linewidth=2)

def sig_line(ax, left, right, ypt, epsilon=.025):
    """ Plot the square brace indicating that two bar plots are significantly
    different
    """
    ax.plot([left, right], [ypt, ypt], color='k', linewidth=3)
    ax.plot([left, left], [ypt-epsilon, ypt], color='k', linewidth=3)
    ax.plot([right, right], [ypt-epsilon, ypt], color='k', linewidth=3)
    ax.text( float(left+right)/2, ypt, '*')

def bar_plot(data, ax=None, sig_pairs=[], ypts=[], 
    colors=['green', 'red', 'magenta', 'blue'], markers=['o', 's', '^', '*'],
    bar_width=0.05, labels=None, show_data=True, data_to_show='pts', 
    fn=np.mean, hide_bar=False, shade_bar=False, start_vals=[], x_offset=0):
    """ Bar plot
    """
    # TODO implement hide_bar, shaded, start_vals
    # TODO multiple bar plots on one axis (i.e. x-offset)
    if ax == None:
        plt.figure()
        ax = plt.subplot(1,1,1)

    elim_ur_lines(ax)
    ax.hold(True)

    if isinstance(data, np.ndarray):
        n_data_bars = data.shape[1]
    elif isinstance(data, list):
        n_data_bars = len(data) 
    
    ticks = []
    for k in range(n_data_bars):
        ctr = bar_width*k
        #print ctr
        #ticks.append(ctr)
        label = None if labels == None else labels[k]
        
        if isinstance(data, np.ndarray):
            _plot_one_bar(
                ax, data[:,k], ctr, width=bar_width, c=colors[k], 
                m=markers[k], label=label, show_data=show_data, 
                data_to_show=data_to_show, fn=fn)
        elif isinstance(data, list):
            _plot_one_bar(
                ax, data[k], ctr, width=bar_width, c=colors[k], 
                m=markers[k], label=label, show_data=show_data,
                data_to_show=data_to_show, fn=fn)
    ax.set_xlim([-bar_width*0.5, (n_data_bars - 0.5)*bar_width])
    ax.set_xticks(ticks)

    for k, pair in enumerate(sig_pairs):
        x, y = pair
        sig_line(ax, ticks[x], ticks[y], ypts[k])
    #return [-bar_width*0.5, (n_data_bars - 0.5)*bar_width]

def write_corner_text(ax, corner, text, size=8):
    """ Write text in one of the corners 'bottom_left', 'bottom_right',
    'top_left', 'top_right'
    """
    #min_x, max_x = ax.get_xlim()
    #min_y, max_y = ax.get_ylim()

    x_left = 0.03
    x_right = 1.
    y_down = 0.03
    y_up = 1.
    #x_left = min_x + 0.03*(max_x-min_x)
    #x_right = max_x
    #y_down = min_y + 0.03*(max_y-min_y)
    #y_up = max_y

    if corner == 'bottom_left':
        ax.text(x_left, y_down, text, 
            horizontalalignment='left', verticalalignment='bottom',
            size=size, transform=ax.transAxes)
    elif corner == 'bottom_right':
        ax.text(x_right, y_down, text, 
            horizontalalignment='right', verticalalignment='bottom',
            size=size, transform=ax.transAxes)
    elif corner == 'top_left':
        ax.text(x_left, y_up, text, 
            horizontalalignment='left', verticalalignment='top',
            size=size, transform=ax.transAxes)
    elif corner == 'top_right':
        ax.text(x_right, y_up, text, 
            horizontalalignment='right', verticalalignment='top',
            size=size, transform=ax.transAxes)

def scatterplot_line(ax, xdata, ydata, xlim, weights, p_corner='top_right'):
    """ 
    Plot linear best-fit for weighted scatterplot data
    """
    inds = ~np.isnan(xdata) * ~np.isnan(ydata)
    coefs = np.polynomial.polynomial.polyfit(xdata[inds], ydata[inds], 
        1, w=weights[inds]/sum(weights[inds]))
    x = np.linspace(xlim[0], xlim[1], 20)
    fit = np.polynomial.polynomial.polyval(x, coefs)
    ax.plot(x, fit, 'k--')
    r, p = pearsonr(xdata, ydata, weights)
    if p < 0.001:
        p_str = r'$r=%0.3g$***' % r 
    elif p < 0.05:
        p_str = r'$r=%0.3g$*' % r
    else:
        p_str = r'$r=%0.3g$' % r 
    write_corner_text(ax, p_corner, p_str, size=8)

def add_cax2(ax, fig, im, cbar_label, lim, fontsize=10):
    cax = add_cax(ax, fig, left_offset=0.03)
    cbar = fig.colorbar(im, cax=cax, ticks=lim)
    ticks = cax.get_ymajorticklabels()
    [tick.set_fontsize(fontsize) for tick in ticks]
    ticks[0].set_va('bottom')
    ticks[1].set_va('top')
    axis_label(cax, cbar_label, offset=2, axis='y', size=fontsize)

def letter_axis(ax, letter, axis_align=0, offset=0):
    print letter
    min_x, max_x = ax.get_xlim()
    min_y, max_y = ax.get_ylim()
    y_pos = max_y + 0.02*(max_y-min_y)
    #ax.text(min_x, y_pos, r"\textbf{%s}" % letter, ha='left', 
    #    va='bottom', size=panel_letter_size)

    return ax.text(axis_align, 1. + offset, r"\textbf{%s}" % letter, ha='right',
        size=panel_letter_size, va='bottom', rotation=0, transform=ax.transAxes)

def letter_axes(axes, **kwargs):
    if isinstance(axes, list):
        [letter_axis(ax, chr(k+65), **kwargs) for k, ax in enumerate(axes)]
    elif isinstance(axes, np.ndarray):
        [letter_axis(ax, chr(k+65), **kwargs) for k, ax in enumerate(axes.ravel())]

def set_xlim(ax, lim=None, labels=[], show_lim=True, size=tick_label_size, axis='x'):
    """ Set x-lim 
    """
    if np.iterable(ax) and np.iterable(show_lim):
        ax_ls = ax
        show_lim_ls = show_lim
        for ax, show_lim in izip(ax_ls, show_lim_ls):
            set_xlim(ax, lim, labels=labels, show_lim=show_lim, size=size, 
            axis=axis)
    elif np.iterable(ax):
        ax_ls = ax
        for ax in ax_ls:
            set_xlim(ax, lim, labels=labels, show_lim=show_lim, size=size, 
            axis=axis)
    else:
        # get axis-dependent functions
        if axis == 'x':
            ticklabel_fn = ax.set_xticklabels
            lim_fn = ax.set_xlim
            tick_fn = ax.set_xticks
            get_ticks = ax.xaxis.get_major_ticks
            tick_alignment = ['left', 'right']
            align_dim = 'set_ha'
            if lim == None: lim = ax.get_xlim()
        elif axis == 'y':
            ticklabel_fn = ax.set_yticklabels
            lim_fn = ax.set_ylim
            tick_fn = ax.set_yticks
            get_ticks = ax.yaxis.get_major_ticks
            tick_alignment = ['bottom', 'top']
            align_dim = 'set_va'
            if lim == None: lim = ax.get_ylim()


        lim_fn(lim)
        tick_fn(lim)
        if len(labels) > 0: 
            ticklabel_fn(labels, size=size)
        elif show_lim:
            ticklabel_fn(lim, size=size)
        else:
            ticklabel_fn([], size=size)
            
        ticks = get_ticks()
        try:
            getattr(ticks[0].label1, align_dim)(tick_alignment[0])
            getattr(ticks[-1].label1, align_dim)(tick_alignment[1])
            #ticks[-1].label1.set_ha(tick_alignment[1])
        except:
            pass

def set_axlim(ax, lim, labels=[], show_lim=True, size=tick_label_size, axis='x'):
    """ Set x-lim 
    """
    if np.iterable(ax) and np.iterable(show_lim):
        ax_ls = ax
        show_lim_ls = show_lim
        for ax, show_lim in izip(ax_ls, show_lim_ls):
            set_xlim(ax, lim, labels=labels, show_lim=show_lim, size=size, 
            axis=axis)
    elif np.iterable(ax):
        ax_ls = ax
        for ax in ax_ls:
            set_xlim(ax, lim, labels=labels, show_lim=show_lim, size=size, 
            axis=axis)
    else:
        # get axis-dependent functions
        if axis == 'x':
            ticklabel_fn = ax.set_xticklabels
            lim_fn = ax.set_xlim
            tick_fn = ax.set_xticks
            get_ticks = ax.xaxis.get_major_ticks
            tick_alignment = ['left', 'right']
            align_dim = 'set_ha'
        elif axis == 'y':
            ticklabel_fn = ax.set_yticklabels
            lim_fn = ax.set_ylim
            tick_fn = ax.set_yticks
            get_ticks = ax.yaxis.get_major_ticks
            tick_alignment = ['bottom', 'top']
            align_dim = 'set_va'


        lim_fn(lim)
        tick_fn(lim)
        if len(labels) > 2: # more labels than just the boundaries
            tick_fn(np.linspace(lim[0], lim[1], len(labels)))
            ticklabel_fn(labels, size=size)
        elif len(labels) > 0: 
            ticklabel_fn(labels, size=size)
        elif show_lim:
            ticklabel_fn(lim, size=size)
        else:
            ticklabel_fn([], size=size)
            
        ticks = get_ticks()
        try:
            getattr(ticks[0].label1, align_dim)(tick_alignment[0])
            getattr(ticks[-1].label1, align_dim)(tick_alignment[1])
            #ticks[-1].label1.set_ha(tick_alignment[1])
        except:
            pass

def set_ylim(ax, lim=None, labels=[], show_lim=True, size=tick_label_size):
    """ 
    Set y-lim 
    """
    ax.set_ylim([lim[0], lim[-1]])
    ax.set_yticks(lim)
    if len(labels) > 0: 
        ax.set_yticklabels(labels, size=size)
    elif show_lim:
        ax.set_yticklabels(lim, size=size)
    else:
        ax.set_yticklabels([], size=size)
        
    ticks = ax.yaxis.get_major_ticks()
    try:
        ticks[0].label1.set_verticalalignment('bottom')
        ticks[-1].label1.set_verticalalignment('top')
    except:
        pass

def set_lim(ax, xlim, ylim):
    set_xlim(ax, xlim, show_lim=True)
    set_ylim(ax, ylim, show_lim=True)

def _gen_rasters(data):
    n_trigs = len(data)

    # vertical position of raster plots for each trial
    vert_pos = (np.arange(n_trigs) + 1)/(n_trigs + 2)

    x_plots = np.zeros([2, 0])
    y_plots = np.zeros([2, 0])

    for row_data, p_lower, p_top in izip(data, vert_pos[:-1], vert_pos[1:]):
        ypos = np.array([p_lower, p_top]).reshape(-1, 1)
        if isinstance(row_data, float) or isinstance(row_data, int):
            n_rasters = 1
        elif np.iterable(row_data):
            n_rasters = len(row_data)
        raster_ypos = np.tile(ypos, [1, n_rasters])
        raster_xpos = np.vstack([row_data, row_data])
        x_plots = np.hstack([x_plots, raster_xpos])
        y_plots = np.hstack([y_plots, raster_ypos])
    return x_plots, y_plots, vert_pos
    
def set_ylim_min(ax, min_lim):
    current_ylim = ax.get_ylim()
    ax.set_ylim( [min_lim, current_ylim[1]] )

def set_labels(ax, xlabel='', ylabel='', size=None):
    if size == None: size=axis_label_size
    ax.set_xlabel(xlabel, size=size)
    ax.set_ylabel(ylabel, size=size)

def set_ylabel(ax, y_label):
    ax.set_ylabel(y_label, size=axis_label_size) 
    
def set_xlabel(ax, x_label):
    ax.set_xlabel(x_label, size=axis_label_size) 

def set_title(ax, title):
    if np.iterable(ax):
        map(lambda axis, axis_title: set_title(axis, axis_title), ax, title)
    else:
        ax.set_title(title, size=title_size)

def clear_ticks(ax):
    """ Helper function to clear axis ticks 
    """
    ax.set_xticks([])
    ax.set_yticks([])

def line_with_error_bar(ax, line, sd, t=None, color='black', alpha=0.15, **kwargs): 
    """
    """
    if t == None: t = np.arange(len(line))
    assert len(line) == len(sd)
    ax.plot(t, line, color=color, **kwargs)
    ax.fill_between(t, line - sd, line + sd, facecolor=color, alpha=alpha)

def subplot_binary_vec(vec, ycoord=0, ax=None, xvec=None, *args, **kwargs):
    """
    ax : axis to plot on
    vec : binary input vector
    """
    vec = vec.copy()
    vec[vec == 0] = np.NaN

    if ax == None:
        plt.figure()
        ax = plt.subplot(1,1,1)

    if xvec == None:
        ax.plot(vec*ycoord, **kwargs) 
    else:
        ax.plot(xvec, vec*ycoord, **kwargs) 

def legend(ax, loc='best'):
    ax.legend(prop={'size':legend_size}, loc=loc)

def save(plot_dir, basename, verbose=False, **kwargs):
    fname = os.path.join(plot_dir, basename)
    if verbose: print fname
    plt.savefig(fname, **kwargs)

def unify_x_axes(axes):
    raise Exception("unify_x_axes incomplete")

def unify_y_axex(axes):
    raise Exception("unify_y_axes incomplete")

def twinx(ax):
    raise Exception("FINISH")
    ax.twinx()
    # TODO move labeling over to the other vertical line (i.e. left ->right)

def elim_ur_lines(ax):
    """ Eliminate the upper right lines from the axis
    """
    elim_lines(ax, ['top', 'right'])
def gen_horiz_axes(n_axes, axis_equal=False, left_offset=0, spacing=0.01, 
    height=0.94, bottom_offset=0.02, sharey=False, scale=1, vert_start=0):
    """

    """
    axes = []
    plot_width = (1 - left_offset)/n_axes
    ystart = scale*((1 - height+bottom_offset)/2 + bottom_offset) + vert_start
    for k in range(n_axes):
        if not axis_equal and (not sharey or k == 0):
            new_ax = plt.axes([left_offset + plot_width*k, ystart, plot_width-spacing, height])
        elif axis_equal and (not sharey or k >0):
            new_ax = plt.axes([left_offset + plot_width*k, ystart, plot_width-spacing, height], aspect=1)
        elif (sharey and k >0):
            new_ax = plt.axes([left_offset + plot_width*k, ystart, plot_width-spacing, height],
                sharey=axes[0])
        axes.append(new_ax)
    return axes

def simplefig(ax):
    """ Eliminate all the axis lines from the ax
    """
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)


def save_and_show(fname, show):
    if fname:
        plt.savefig(fname)

    if show:
        plt.show()

def axis_label(ax, label, offset=-0.1, axis_align=0.5, size=10, 
    axis='x'):
    """
    """
    if axis == 'y':
        return ax.text(offset, axis_align, label, ha='center', 
            size=size, va='center', rotation=90, transform=ax.transAxes)
    elif axis == 'x':
        return ax.text(axis_align, offset, label, ha='center', 
            size=size, va='center', rotation=0, transform=ax.transAxes)

def xlabel(ax, label, **kwargs):
    kwargs['offset'] = kwargs.pop('offset', -0.05)
    if np.iterable(ax):
        map(lambda x: xlabel(x, label, **kwargs), ax)
    else:
        kwargs.pop('axis', 'x')
        return axis_label(ax, label, axis='x', **kwargs)

def ylabel(ax, label, **kwargs):
    kwargs['offset'] = kwargs.pop('offset', -0.03)
    if np.iterable(ax):
        map(lambda x: ylabel(x, label, **kwargs), ax)
    else:
        kwargs.pop('axis', 'y')
        return axis_label(ax, label, axis='y', **kwargs)

def ylabel_widths(axes):
    n_cols = axes.shape[1]
    n_rows = axes.shape[0]
    max_tick_widths = np.zeros([n_rows, n_cols])
    for k in range(n_cols):
        for m in range(n_rows):
            ax = axes[m,k]
            ticks = ax.yaxis.get_major_ticks()
            tick_widths = np.zeros(len(ticks))
            for j, tick in enumerate(ticks):
                # TODO exclude invisible ticks
                label = tick.label
                label_extent = label.get_window_extent()
                fig_extent = fig.get_window_extent()
                
                label_display_width = label_extent.x1 - label_extent.x0
                label_display_height = label_extent.y1 - label_extent.y0
                
                fig_display_width = fig_extent.x1 - fig_extent.x0
                fig_display_height = fig_extent.y1 - fig_extent.y0
                tick_widths[j] = label_display_width / fig_display_width
            max_tick_widths[m,k] = max(tick_widths)
    return max_tick_widths            

def histogram_line(ax, data, bins, normed=True, **plot_kwargs):
    hist_data, _ = np.histogram(data, bins, normed=normed)
    bins = bins[:-1] + bins[1]/2
    ax.plot(bins, hist_data, **plot_kwargs)
    return hist_data
