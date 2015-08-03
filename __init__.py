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
import matplotlib.font_manager as fm
import matplotlib.patches as patches
from itertools import izip
import numpy as np
import os.path as path
from pylab import Circle
import os
from collections import OrderedDict
from scipy import stats

default_font = fm.FontProperties(fname='Helvetica.ttf')
bold_font = fm.FontProperties(fname='HelveticaBold.ttf')

ieee_fig_font = fm.FontProperties(fname='Helvetica.ttf', size=8)
ppt_font = fm.FontProperties(fname='Helvetica.ttf', size=18)
legend_font = fm.FontProperties(fname='Helvetica.ttf', size=8)

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

## Declare the "regular" font
from matplotlib.font_manager import FontProperties
reg_font = FontProperties()

color_names = colors.keys()
n_colors = len(color_names)

## Constants 
panel_letter_size = 12
title_size = 8
tick_label_size = 8
axis_label_size = 8
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


def label(ax, pt, text):
    raise Exception("FINISH")

## Tools for manipulating matplotlib figures 
def subplots(n_vert_plots, n_horiz_plots, x=0.03, y=0.05, left_offset=0.06, right_offset=0.,
    aspect=None, top_offset=0, bottom_offset=0.075, return_flat=False, show_ur_lines=False,
    hold=True, ylabels=False, sep_ylabels=False, xlabels=False, 
    sep_xlabels=False, letter_offset=None, fig=None, **kwargs):
    """
    An implementation of the subplots function

    Parameters
    ==========

    Returns
    =======
    """
    if bottom_offset == None:
        bottom_offset = left_offset
    vert_frac_per_row = (1.-bottom_offset-top_offset)/n_vert_plots
    horiz_frac_per_col = (1.-left_offset-right_offset)/n_horiz_plots
    subplot_width = horiz_frac_per_col - x 
    subplot_height = vert_frac_per_row - y

    if fig is not None:
        fn = fig.add_axes
    else:
        fn = plt.axes

    axes = []
    for m in range(n_vert_plots):
        axes_row = []
        for n in range(n_horiz_plots):
            xstart = left_offset + x/2 + horiz_frac_per_col*n 
            ystart = bottom_offset + y/2 + (n_vert_plots - 1 - m)*vert_frac_per_row
            if aspect is not None:
                new_ax = fn([xstart, ystart, subplot_width, subplot_height], aspect=aspect, **kwargs)
            else:
                new_ax = fn([xstart, ystart, subplot_width, subplot_height], **kwargs)

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

text_height = 0.1 # inches
inches_per_point = 0.0138888889
inches_per_newline = 0.0437391513912007


def subplots2(n_vert_plots, n_horiz_plots, x=0.03, y=1, left_offset=0, right_offset=None,
    top_offset=1, bottom_offset=1, return_flat=False, left_fig_offset_frac=0, right_fig_offset_frac=0, 
    top_fig_frac_offset=0, bottom_fig_frac_offset=0, border=0, hold=True, fig=None, font=ieee_fig_font, sharex=False, **kwargs):
    """
    An implementation of the subplots function

    Parameters
    ==========

    Returns
    =======
    """    
    if fig == None:
        fig = plt.gcf()
    fig_width_inches = fig.bbox._bbox.x1 - fig.bbox._bbox.x0
    fig_height_inches = fig.bbox._bbox.y1 - fig.bbox._bbox.y0


    left_fig_offset_frac = max(left_fig_offset_frac, border)
    right_fig_offset_frac = max(right_fig_offset_frac, border)
    top_fig_frac_offset = max(top_fig_frac_offset, border)
    bottom_fig_frac_offset = max(bottom_fig_frac_offset, border)


    try:
        text_height_ = text_height_inches('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVXYZ', fontproperties=font)
    except RuntimeError:
        text_height_ = text_height


    top_offset = top_fig_frac_offset + (4*inches_per_point + top_offset*text_height_ + (top_offset-1)*inches_per_newline)/fig_height_inches
    bottom_offset = bottom_fig_frac_offset + (4*inches_per_point + bottom_offset*text_height_ + (bottom_offset-1)*inches_per_newline)/fig_height_inches
    left_offset = left_fig_offset_frac + x + (4*inches_per_point + left_offset*text_height_ + (left_offset-1)*inches_per_newline)/fig_width_inches
    if right_offset == None:
        right_offset = right_fig_offset_frac + 4*inches_per_point/fig_width_inches
    y = (4*inches_per_point + 4*inches_per_point + y*text_height_ + (y-1)*inches_per_newline)/fig_height_inches

    vert_frac_per_row = (1.-bottom_offset-top_offset - y*(n_vert_plots-1))/n_vert_plots
    horiz_frac_per_col = (1.-left_offset-right_offset - x*(n_horiz_plots-1))/n_horiz_plots
    subplot_width = horiz_frac_per_col
    subplot_height = vert_frac_per_row

    if fig is not None:
        fn = fig.add_axes
    else:
        fn = plt.axes

    axes = []
    for m in range(n_vert_plots):
        axes_row = []
        for n in range(n_horiz_plots):
            xstart = left_offset + horiz_frac_per_col*n + n*x
            ystart = bottom_offset + (n_vert_plots - 1 - m)*(vert_frac_per_row + y)

            if m == 0 and n > 0 and sharex:
                new_ax = fn([xstart, ystart, subplot_width, subplot_height], sharex=axes_row[0], **kwargs)
            elif (m > 0 or n > 0) and sharex:
                new_ax = fn([xstart, ystart, subplot_width, subplot_height], sharex=axes[0][0], **kwargs)
            else:
                new_ax = fn([xstart, ystart, subplot_width, subplot_height], **kwargs)


            elim_ur_lines(new_ax)

            # set hold property for ax
            new_ax.hold(hold)

            axes_row.append(new_ax)
        axes.append(axes_row)

    axes = np.array(axes)

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
    label = kwargs.pop('label', None)
    ax.plot(x_data, y_data, color=color, label=label, **kwargs)
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
    if np.iterable(ax):
        map(lambda x: set_ticklabel_size(x, xsize=xsize, ysize=ysize), ax)
        return 
        
    if ysize == None:
        ysize = xsize
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_font_properties(ieee_fig_font)
        # tick.label.set_fontsize(xsize)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_font_properties(ieee_fig_font)
        # tick.label.set_fontsize(ysize)

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
    label=None, show_data=True, data_to_show='pts', edgecolor='none', facecolor='none', linewidth=3, **kwargs):
    """ Generate a single bar plot
    """ 
    if edgecolor == 'none':
        edgecolor = c
    n_data_pts = len(data)
    ax.bar(ctr-width/4, fn(data), width=width/2, color=facecolor, edgecolor='black', facecolor=c,
        linewidth=linewidth, label=label, hatch=kwargs.pop('hatch', None))
    if show_data:
        if data_to_show == 'pts':
            ax.scatter(np.ones(n_data_pts)*ctr, data, c=c, s=s, marker=m)
        elif data_to_show == 'std':
            std = np.std(data)
            ax.plot([ctr, ctr], [fn(data)-std, fn(data)+std], color='black', linewidth=2)
        elif data_to_show == 'sem':
            from scipy.stats import sem
            std = sem(data)
            ax.plot([ctr, ctr], [fn(data)-std, fn(data)+std], color='black', linewidth=2)            

def plot_asterisks(ax, x, y, text, pt_offset=2, ycoords='axis', **kwargs):
    if ycoords == 'axis':
        trans = ax.get_xaxis_transform()
        rel_offset = pt_offset*inches_per_point/_get_ax_height_inches(ax)
        ax.text(x, y + rel_offset, text, ha='center', va='bottom', size=8, transform=trans, **kwargs)        
    elif ycoords == 'data':
        data_units_per_inch = np.diff(ax.get_ylim())[0] / _get_ax_height_inches(ax)
        rel_offset = pt_offset*inches_per_point * data_units_per_inch
        ax.text(x, y + rel_offset, text, ha='center', va='bottom', size=8, **kwargs)        


def sig_line(ax, left, right, text, ypt=0.9, pt_offset=0):
    """ 
    Plot the square brace indicating that two bar plots are significantly
    different
    """
    # TODO the 1.02 factor should be (optimally) specified as 4 "points", not as a fraction of the size of the axis!
    trans = ax.get_xaxis_transform()
    plot_asterisks(ax, float(left+right)/2, ypt, text, pt_offset=pt_offset)
    # rel_offset = 2*inches_per_point/_get_ax_height_inches(ax)
    # ax.text(float(left+right)/2, ypt + rel_offset, text, ha='center', va='bottom', size=8, transform=trans)
    ylim = ax.get_ylim()
    ax.plot([left, right], [ypt, ypt], color='black', transform=trans)
    ax.set_ylim(ylim)

def bar_plot_test(data_a, data_b, ax, **kwargs):
    bar_plot([data_a, data_b], ax=ax, data_to_show='sem', **kwargs)
    from scipy.stats import kruskal
    _, p = kruskal(data_a, data_b)
    ast = p_val_to_asterisks(p)
    sig_line(ax, 0, 0.05, ast, pt_offset=0)

def bar_plot(data, ax=None, sig_pairs=[], ypts=[], 
    colors=['green', 'red', 'magenta', 'blue'], markers=['o', 's', '^', '*'],
    bar_width=0.05, labels=None, show_data=True, data_to_show='pts', 
    fn=np.mean, hide_bar=False, shade_bar=False, start_vals=[], x_offset=0, hatching=None, **kwargs):
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

        if hatching is not None:
            hatch = hatching[k]
        else:
            hatch = None
        
        if isinstance(data, np.ndarray):
            _plot_one_bar(
                ax, data[:,k], ctr, width=bar_width, c=colors[k], 
                m=markers[k], label=label, show_data=show_data, 
                data_to_show=data_to_show, fn=fn, hatch=hatch, **kwargs)
        elif isinstance(data, list):
            _plot_one_bar(
                ax, data[k], ctr, width=bar_width, c=colors[k], 
                m=markers[k], label=label, show_data=show_data,
                data_to_show=data_to_show, fn=fn, hatch=hatch, **kwargs)
    ax.set_xlim([-bar_width*0.5, (n_data_bars - 0.5)*bar_width])
    ax.set_xticks(ticks)

    for k, pair in enumerate(sig_pairs):
        x, y = pair
        sig_line(ax, ticks[x], ticks[y], ypts[k])
    return ticks
    #return [-bar_width*0.5, (n_data_bars - 0.5)*bar_width]

def write_corner_text(ax, corner, text, fontproperties=ieee_fig_font, **kwargs):
    """ Write text in one of the corners 'bottom_left', 'bottom_right',
    'top_left', 'top_right'
    """
    x_left = 0.03
    x_right = 1.
    y_down = 0.03
    y_up = 1.


    if isinstance(corner, tuple):
        ax.text(corner[0], corner[1], text, 
            horizontalalignment='left', verticalalignment='bottom',
            fontproperties=fontproperties, transform=ax.transAxes, **kwargs)

    if corner == 'bottom_left':
        ax.text(x_left, y_down, text, 
            horizontalalignment='left', verticalalignment='bottom',
            fontproperties=fontproperties, transform=ax.transAxes, **kwargs)
    elif corner == 'bottom_right':
        ax.text(x_right, y_down, text, 
            horizontalalignment='right', verticalalignment='bottom',
            fontproperties=fontproperties, transform=ax.transAxes, **kwargs)
    elif corner == 'top_left':
        ax.text(x_left, y_up, text, 
            horizontalalignment='left', verticalalignment='top',
            fontproperties=fontproperties, transform=ax.transAxes, **kwargs)
    elif corner == 'top_right':
        ax.text(x_right, y_up, text, 
            horizontalalignment='right', verticalalignment='top',
            fontproperties=fontproperties, transform=ax.transAxes, **kwargs)

def pearsonr(x, y, weights=None):
    x = np.array(x)
    y = np.array(y)
    assert len(x) == len(y)
    N = len(x)
    if weights == None:
        weights = np.ones(len(x))*1./N
    else:
        weights = np.array(weights, dtype=np.float64)

    weights = np.array(weights, dtype=np.float64)/ np.sum(weights)

    inds = np.logical_and(~np.isnan(x), ~np.isnan(y))
    x = x[inds]
    y = y[inds]
    weights = weights[inds]
    
    mean_x = np.sum(x*weights)
    mean_y = np.sum(y*weights)
    cov_est = sum( weights*(x-mean_x)*(y-mean_y) )
    std_x = np.sqrt( np.sum( weights*(x-mean_x)**2 ) )
    std_y = np.sqrt( np.sum( weights*(y-mean_y)**2 ) )
    r = cov_est/(std_x*std_y)

    # calculate p value
    df = N-2
    t = r*np.sqrt( float(df)/(1-r**2) )
    
    p = 2*(stats.t.cdf( -abs(t), df))

    return r, p 

def scatter(ax, x, y=None, **kwargs):
    if y == None:
        y = x
        x = np.arange(len(y))
    ax.scatter(x, y, **kwargs)

def clean_up_ticks(axes, font=ieee_fig_font):
    if np.iterable(axes):
        for ax in axes.ravel():
            clean_up_ticks(ax, font=font)
        
    else:
        set_axlim(axes, axis='x')
        set_axlim(axes, axis='y')
        set_ticklabel_size(axes, xsize=8, ysize=8)

        ax = axes
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_font_properties(font)
            
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_font_properties(font)        

def p_val_to_asterisks(p):
    if p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    else:
        return 'ns'

def scatterplot_line(ax, xdata, ydata=None, eval_data=None, xlim=None, weights=None, p_corner='top_right', color='blue', 
    text_kwargs=dict(size=8), sided=0, verbose=False, write_rsq=False, **kwargs):
    """ 
    Plot linear best-fit for weighted scatterplot data
    """
    if ydata is None:
        ydata = xdata
        xdata = np.arange(len(ydata))
    inds = ~np.isnan(xdata) * ~np.isnan(ydata)
    if weights == None:
        coefs = np.polynomial.polynomial.polyfit(xdata[inds], ydata[inds], 1)
    else:
        coefs = np.polynomial.polynomial.polyfit(xdata[inds], ydata[inds], 
            1, w=weights[inds]/sum(weights[inds]))
    if xlim is None:
        x = xdata
    else:
        x = np.linspace(xlim[0], xlim[1], 20)

    inds = np.argsort(x)
    if eval_data == None:
        eval_data = x[inds]
    fit = np.polynomial.polynomial.polyval(eval_data, coefs)

    ax.plot(eval_data, fit, color=color, **kwargs)
    r, p = pearsonr(xdata, ydata, weights)

    if sided * r > 0:
        p = p/2

    if p < 0.001:
        p_str = r'$r=%0.3g$***' % r if not write_rsq else r'$r^2=%0.3g$***' % r**2 
    if p < 0.01:
        p_str = r'$r=%0.3g$**' % r if not write_rsq else r'$r^2=%0.3g$**' % r**2 
    elif p < 0.05:
        p_str = r'$r=%0.3g$*' % r if not write_rsq else r'$r^2=%0.3g$*' % r**2 
    elif verbose:
        p_str = r'$r=%0.3g, p=%0.2g$' % (r, p) 
    else:
        p_str = r'$r=%0.3g$' % (r,)

    if not p_corner == '':
        write_corner_text(ax, p_corner, p_str, **text_kwargs)


def add_cax2(ax, fig, im, cbar_label, lim=None, font=ieee_fig_font, left_offset=0.03, axis_label_offset=3, **kwargs):
    """
    Add colorbar axis to the right of an axis 
    """
    bbox = ax.get_position()
    cax = fig.add_axes([bbox.x1+left_offset, bbox.y0, 0.03, bbox.y1-bbox.y0])    
    cbar = fig.colorbar(im, cax=cax, **kwargs)
    
    ticks = cax.get_ymajorticklabels()
    [tick.set_font_properties(font) for tick in ticks]
    ticks[0].set_va('bottom')
    ticks[1].set_va('top')
    caxis_label(cax, cbar_label, axis='y', line_offset=-0.5, fontproperties=font)
    return cax, cbar

add_cax = add_cax2

def letter_axis(ax, letter, axis_align=0, offset=0, va='bottom', ha='right', **kwargs):
    min_x, max_x = ax.get_xlim()
    min_y, max_y = ax.get_ylim()
    y_pos = max_y + 0.02*(max_y-min_y)

    return ax.text(axis_align, 1. + offset, str(letter), ha=ha,
        size=panel_letter_size, va=va, rotation=0, transform=ax.transAxes, fontproperties=bold_font)

def letter_row(axes, first_letter='A', pt_offset=4, xfrac=0.5):
    axes = axes.ravel()
    axes_x_extent = np.vstack([(ax.bbox._bbox.x0, ax.bbox._bbox.x1) for ax in axes])
    axes_x_space_fig_frac = np.max(axes_x_extent[1:,0] - axes_x_extent[:-1, 1])

    axes_y_extent = np.vstack([(ax.bbox._bbox.y0, ax.bbox._bbox.y1) for ax in axes])
    axes_y_space_fig_frac = np.max(axes_y_extent[1:,0] - axes_y_extent[:-1, 1])

    fig = axes[0].figure
    n_axes = len(axes)
    for k in range(n_axes):
        letter = chr(k + ord(first_letter))
        fig_height_inches = fig.bbox._bbox.y1 - fig.bbox._bbox.y0
        offset = (text_height/2 + pt_offset*inches_per_point)/fig_height_inches
        fig.text(axes_x_extent[k,0] - xfrac*axes_x_space_fig_frac, axes_y_extent[k,1] + offset, letter, ha='left', va='bottom', size=12, fontproperties=bold_font)
    return axes_x_space_fig_frac/2, 0.98

def letter_col(axes, first_letter='A', pt_offset=4, yfrac=0.0):
    axes = axes.ravel()
    axes_x_extent = np.vstack([(ax.bbox._bbox.x0, ax.bbox._bbox.x1) for ax in axes])
    axes_x_space_fig_frac = np.max(axes_x_extent[1:,0] - axes_x_extent[:-1, 1])

    axes_y_extent = np.vstack([(ax.bbox._bbox.y0, ax.bbox._bbox.y1) for ax in axes])
    axes_y_space_fig_frac = np.max(axes_y_extent[:-1, 1] - axes_y_extent[1:,1])

    print axes_y_extent
    print axes_y_space_fig_frac

    fig = axes[0].figure
    n_axes = len(axes)
    for k in range(n_axes):
        letter = chr(k + ord(first_letter))
        fig_width_inches = fig.bbox._bbox.x1 - fig.bbox._bbox.x0
        offset = (text_height/2 + pt_offset*inches_per_point)/fig_width_inches
        fig.text(0.5*offset, axes_y_extent[k,1] + yfrac*axes_y_space_fig_frac, letter, ha='left', va='bottom', size=12, fontproperties=bold_font)
    return axes_x_space_fig_frac/2, 0.98

def title_multi_axis(axes, title, line_offset=1):
    axes = axes.ravel()
    axes_x_extent = np.vstack([(ax.bbox._bbox.x0, ax.bbox._bbox.x1) for ax in axes])
    axes_x_space_fig_frac = np.max(axes_x_extent[1:,0] - axes_x_extent[:-1, 1])
    x_min = min(axes_x_extent.ravel())
    x_max = max(axes_x_extent.ravel())
    x = (x_min + x_max)/2
        
    axes_y_extent = np.vstack([(ax.bbox._bbox.y0, ax.bbox._bbox.y1) for ax in axes])
    # axes_y_space_fig_frac = np.max(axes_y_extent[1:,0] - axes_y_extent[:-1, 1])

    try:
        text_height_ = text_height_inches(title)
    except RuntimeError:
        text_height_ = text_height

    fig = axes[0].figure
    dist_from_axis = (4*inches_per_point + line_offset*(inches_per_newline+text_height_))/_get_fig_height_inches(fig) 
    fig.text(x, axes_y_extent[0,1] + dist_from_axis, title, ha='center', va='bottom', size=8, fontproperties=ieee_fig_font)

def letter_axes(axes, **kwargs):
    # TODO determine the left starting point of each label relative to the 
    # axis by determining the spacing between the axes 
    if isinstance(axes, list):
        [letter_axis(ax, chr(k+65), **kwargs) for k, ax in enumerate(axes)]
    elif isinstance(axes, np.ndarray):
        [letter_axis(ax, chr(k+65), **kwargs) for k, ax in enumerate(axes.ravel())]

def set_xlim(ax, lim=None, labels=[], show_lim=True, size=tick_label_size, axis='x',):
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
        except:
            pass

def set_axlim(ax, lim=None, labels=[], show_lim=True, size=tick_label_size, axis='x', realign_ticks=False, font=ieee_fig_font):
    """ 
    Set x-lim 
    """
    if np.iterable(ax) and np.iterable(show_lim):
        ax_ls = ax
        show_lim_ls = show_lim
        for ax, show_lim in izip(ax_ls, show_lim_ls):
            set_axlim(ax, lim, labels=labels, show_lim=show_lim, font=font, 
                      axis=axis, realign_ticks=realign_ticks)
    elif np.iterable(ax):
        ax_ls = ax
        for ax in ax_ls:
            set_axlim(ax, lim, labels=labels, show_lim=show_lim, font=font, 
                      axis=axis, realign_ticks=realign_ticks)
    else:
        # if no limits are specified, then just use the current axis limits already in the plot
        if lim == None:
            lim = getattr(ax, 'get_%slim' % axis)()

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
        elif axis == 'z':
            ticklabel_fn = ax.set_zticklabels
            lim_fn = ax.set_zlim
            tick_fn = ax.set_zticks
            get_ticks = ax.zaxis.get_major_ticks
            tick_alignment = ['bottom', 'top']
            align_dim = 'set_va'            


        lim_fn([lim[0], lim[-1]])
        tick_fn(lim)
        if len(labels) > len(lim):
            tick_fn(labels)
            ticklabel_fn(labels, fontproperties=font)
        elif len(labels) > 0: 
            ticklabel_fn(labels, fontproperties=font)
        elif show_lim:
            ticklabel_fn(lim, fontproperties=font)
        else:
            ticklabel_fn([], fontproperties=font)
            
        ticks = get_ticks()
        try:
            if realign_ticks:
                getattr(ticks[0].label1, align_dim)(tick_alignment[0])
                getattr(ticks[-1].label1, align_dim)(tick_alignment[1])
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

def set_xlim_min(ax, min_lim):
    current_xlim = ax.get_xlim()
    ax.set_xlim( [min_lim, current_xlim[1]] )    

def set_labels(ax, xlabel='', ylabel='', size=None):
    if size == None: size=axis_label_size
    ax.set_xlabel(xlabel, size=size)
    ax.set_ylabel(ylabel, size=size)

def set_ylabel(ax, y_label):
    ax.set_ylabel(y_label, size=axis_label_size) 
    
def set_xlabel(ax, x_label):
    ax.set_xlabel(x_label, size=axis_label_size) 

def set_title(ax, title, **kwargs):
    if np.iterable(ax):
        map(lambda axis, axis_title: set_title(axis, axis_title, **kwargs), ax, title)
    else:
        if 'fontproperties' not in kwargs:
            kwargs['fontproperties'] = ieee_fig_font
        line_offset = kwargs.pop('line_offset', 0)
        axis_height_inches = _get_ax_height_inches(ax)

        try:
            text_height_ = text_height_inches(title)
        except RuntimeError:
            text_height_ = text_height

        dist_from_axis = (4*inches_per_point + line_offset*(inches_per_newline+text_height_))/axis_height_inches        
        ax.text(0.5, 1 + dist_from_axis, title, ha='center', va='bottom', rotation=0, transform=ax.transAxes, **kwargs)

def _get_ax_height_inches(ax):
    fig = ax.figure
    fig_width_inches = fig.bbox._bbox.x1 - fig.bbox._bbox.x0
    fig_height_inches = fig.bbox._bbox.y1 - fig.bbox._bbox.y0

    axis_window_extent = ax.get_window_extent()._bbox
    axis_yextent_rel = axis_window_extent.y1 - axis_window_extent.y0
    axis_height_inches = fig_height_inches * axis_yextent_rel
    return axis_height_inches    

def _get_ax_width_inches(ax):
    fig = ax.figure
    fig_width_inches = fig.bbox._bbox.x1 - fig.bbox._bbox.x0
    fig_height_inches = fig.bbox._bbox.y1 - fig.bbox._bbox.y0

    axis_window_extent = ax.get_window_extent()._bbox
    axis_xextent_rel = axis_window_extent.x1 - axis_window_extent.x0
    axis_height_inches = fig_width_inches * axis_xextent_rel
    return axis_height_inches

def _get_fig_height_inches(fig):
    fig_height_inches = fig.bbox._bbox.y1 - fig.bbox._bbox.y0
    return fig_height_inches

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

def legend(ax, loc='best', size=legend_size, edgecolor='white', **kwargs):
    legend_font = fm.FontProperties(fname='/Users/sgowda/code/plotutil/Helvetica.ttf', size=size)
    leg = ax.legend(prop=legend_font, loc=loc, **kwargs)
    leg.get_frame().set_edgecolor(edgecolor) 

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

def text_height_inches(label, fontproperties=ieee_fig_font, **kwargs):
    test_fig = plt.figure()    
    test_ax = plt.subplot(111)

    label_obj = test_ax.text(0.5, 0.5, label, rotation=0, transform=test_ax.transAxes, fontproperties=fontproperties, **kwargs)

    plt.show()
    try:
        bbox = label_obj.get_window_extent()
        return (bbox.y1 - bbox.y0)/test_fig.dpi
    except:
        return text_height
    finally:
        plt.close(test_fig)
        
    


def axis_label(ax, label, offset=None, axis_align=0.5, size=8, 
    axis='x', line_offset=0, fontproperties=ieee_fig_font, **kwargs):
    """
    """
    kwargs['multialignment'] = kwargs.pop('multialignment', 'center')

    try:
        text_height_ = text_height_inches('aA', fontproperties=fontproperties, **kwargs)
    except RuntimeError:
        text_height_ = text_height

    if offset == None and axis == 'x':
        axis_height_inches = _get_ax_height_inches(ax)
        dist_from_axis = (-4*inches_per_point - line_offset*(inches_per_newline+text_height_))/axis_height_inches
    elif offset == None and axis == 'y':
        axis_width_inches = _get_ax_width_inches(ax)
        dist_from_axis = (-4*inches_per_point - line_offset*(inches_per_newline+text_height_))/axis_width_inches
    else:
        dist_from_axis = offset

    if axis == 'y':
        return ax.text(dist_from_axis, 0.5, label, ha='right', va='center', rotation=90, transform=ax.transAxes, fontproperties=fontproperties, **kwargs)
    elif axis == 'x':
        return ax.text(0.5, dist_from_axis, label, ha='center', va='top', rotation=0, transform=ax.transAxes, fontproperties=fontproperties, **kwargs)

def caxis_label(ax, label, offset=None, axis_align=0.5, size=8, 
    axis='x', line_offset=0, fontproperties=ieee_fig_font, **kwargs):
    """
    this function is the same as the axis_label function, but the axis is on the opposite sides
    """
    kwargs['multialignment'] = kwargs.pop('multialignment', 'center')

    try:
        text_height_ = text_height_inches(label, fontproperties=fontproperties, **kwargs)
    except RuntimeError:
        text_height_ = text_height

    if offset == None and axis == 'x':
        axis_height_inches = _get_ax_height_inches(ax)
        dist_from_axis = (-4*inches_per_point - line_offset*(inches_per_newline+text_height_))/axis_height_inches
    elif offset == None and axis == 'y':
        axis_width_inches = _get_ax_width_inches(ax)
        dist_from_axis = (-4*inches_per_point - line_offset*(inches_per_newline+text_height_))/axis_width_inches
    else:
        dist_from_axis = offset

    if axis == 'y':
        return ax.text(dist_from_axis, 0.5, label, ha='left', va='center', rotation=90, transform=ax.transAxes, fontproperties=fontproperties, **kwargs)
    elif axis == 'x':
        print dist_from_axis
        return ax.text(0.5, dist_from_axis, label, ha='center', va='bottom', rotation=0, transform=ax.transAxes, fontproperties=fontproperties, **kwargs)


def xlabel(ax, label, **kwargs):
    # kwargs['offset'] = kwargs.pop('offset', -0.1)
    if np.iterable(ax):
        map(lambda x: xlabel(x, label, **kwargs), ax)
    else:
        kwargs.pop('axis', 'x')
        return axis_label(ax, label, axis='x', **kwargs)

def ylabel(ax, label, **kwargs):
    # kwargs['offset'] = kwargs.pop('offset', -0.03)
    if np.iterable(ax):
        map(lambda x: ylabel(x, label, **kwargs), ax)
    else:
        kwargs.pop('axis', 'y')
        return axis_label(ax, label, axis='y', **kwargs)

def histogram2d(xdata, ydata, ax=None, bins=30):
    # Estimate the 2D histogram
    H, xedges, yedges = np.histogram2d(xdata, ydata, bins=bins)
    
    # H needs to be rotated and flipped
    H = np.rot90(H)
    H = np.flipud(H)
     
    # Mask zeros
    Hmasked = np.ma.masked_where(H==0, H) # Mask pixels with a value of zero
     
    # Plot 2D histogram using pcolor
    if ax is None:
        fig2 = plt.figure()
        ax = plt.subplot(111)
    ax.pcolormesh(xedges,yedges,Hmasked)

def row_label(ax, label, **kwargs):
    '''
    For now, this function is the same as ylabel. Eventually it should be able to use renderers to automatically determine the correct offset..
    '''
    if 'offset' in kwargs:
        kwargs['offset'] = -np.abs(kwargs['offset'])
    ylabel(ax, label, **kwargs)

def set_symmetric_ylim(ax, **kwargs):
    m = np.max(np.abs(ax.get_ylim()))
    set_axlim(ax, [-m, m], axis='y', **kwargs)

def multi_axis_row_label(axes, label, offset=0, **kwargs):
    # offset = -np.abs(offset)
    y_max = np.max([ax.bbox._bbox.y1 for ax in axes.ravel()])
    y_min = np.min([ax.bbox._bbox.y0 for ax in axes.ravel()])

    fig_y_coord = (y_max + y_min)/2
    print y_max, y_min, fig_y_coord
    fig = axes.ravel()[0].figure
    fig.text(offset, fig_y_coord, label, rotation=90, ha='right', va='center', fontproperties=ieee_fig_font, **kwargs)

def multi_axis_col_label(axes, label, offset=0, **kwargs):
    x_max = np.max([ax.bbox._bbox.x1 for ax in axes.ravel()])
    x_min = np.min([ax.bbox._bbox.x0 for ax in axes.ravel()])

    fig_x_coord = (x_max + x_min)/2
    fig = axes.ravel()[0].figure
    fig.text(fig_x_coord, offset, label, rotation=0, ha='center', va='bottom', fontproperties=ieee_fig_font, **kwargs)

def _yticklabel_widths(ax, relative_to='figure'):
    fig = ax.figure
    ticks = ax.yaxis.get_major_ticks()
    opt_spacing = []
    for tick in ticks:
        if not tick.get_visible():
            continue
        tick_extent = tick.label.get_window_extent()
        pixel_width = tick_extent.x1 - tick_extent.x0
        inch_width = pixel_width/fig.dpi
        fig_width = ax.figure.bbox._bbox.x1 - ax.figure.bbox._bbox.x0

        if relative_to == 'figure':
            opt_spacing.append((inch_width + 8.*inches_per_point)/fig_width)
        elif relative_to == 'axis':
            opt_spacing.append((inch_width + 8.*inches_per_point)/_get_ax_width_inches(ax))
        elif relative_to == 'inches':
            opt_spacing.append(inch_width + 8.*inches_per_point)
    return opt_spacing

def _xticklabel_heights(ax, relative_to='figure'):
    fig = ax.figure
    ticks = ax.xaxis.get_major_ticks()
    opt_spacing = []
    for tick in ticks:
        if not tick.get_visible():
            continue        
        tick_extent = tick.label.get_window_extent()
        pixel_height = tick_extent.y1 - tick_extent.y0
        inch_height = pixel_height/fig.dpi
        fig_height = ax.figure.bbox._bbox.y1 - ax.figure.bbox._bbox.y0
        if relative_to == 'figure':
            opt_spacing.append((inch_height + 8*inches_per_point)/fig_height)
        elif relative_to == 'axis':
            opt_spacing.append((inch_height + 8*inches_per_point)/_get_ax_height_inches(ax))
    return opt_spacing 

def ylabel_widths(axes):
    fig = axes[0,0].figure
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
    bins = bins[:-1]# + bins[1]/2
    ax.plot(bins, hist_data, **plot_kwargs)
    return hist_data

def calc_row_label_offset(axes):
    return np.max(map(lambda ax: _yticklabel_widths(ax, relative_to='axis'), axes[:,0]))

