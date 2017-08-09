#!/usr/bin/env python
"""

Last Updated:
"""

# Standard modules
import numpy as np
import sys
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib.patches as patches
from scipy.interpolate import interp1d
from matplotlib._png import read_png
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
# mvanvleet specific modules
#from chemistry import io

###########################################################################
####################### Global Variables ##################################
error_message='''
---------------------------------------------------------------------------
Improperly formatted arguments. Proper usage is as follows:

$ 

(<...> indicates required argument, [...] indicates optional argument)
---------------------------------------------------------------------------
    '''



###########################################################################
###########################################################################


###########################################################################
######################## Command Line Arguments ###########################

exp1_file = 'harvey_experimental_virials.dat'
#exp2_file = 'duska_experimental_virials.dat'
exp2_file = 'mbpol_classical.dat'

plot_exp2 = False

#mastiff_file = '01_mastiff_h2o/mastiff_virials.dat'
mastiff_file = 'mastiff_virials.dat'
isaff_file = 'isaff_virials.dat'
isoff_file = 'isoff_virials.dat'
# saptff2_file = 'ethane_ethane_saptff_jpc_2013/jpc_saptff_2nd_virial.dat'

# Experimental equation of state from https://doi.org/10.1006/jcht.1998.0359
#experimental_eos = lambda T: 257.6 - 161.8*np.exp(416.8/T)

###########################################################################
###########################################################################


###########################################################################
########################## Main Code ######################################

# Read data from each file
exp1_data = pd.read_csv(
                exp1_file,delim_whitespace=True,
                names=['T','B'],skiprows=1,
                comment='#')

exp2_data = pd.read_csv(
                exp2_file,delim_whitespace=True,
                names=['T','B','dB'],skiprows=1,
                comment='#')

mastiff_data = pd.read_csv(
                mastiff_file,delim_whitespace=True,
                names=['T','B'],skiprows=5,
                comment='#')

isaff_data = pd.read_csv(
                isaff_file,delim_whitespace=True,
                names=['T','B'],skiprows=5,
                comment='#')

isoff_data = pd.read_csv(
                isoff_file,delim_whitespace=True,
                names=['T','B'],skiprows=5,
                comment='#')


# Set some global color preferences for how the graph's colors should look
sns.set_context('talk',font_scale=2.0)
sns.set_palette('Set2')
sns.set_style("ticks", 
#sns.set_style("ticks", 
                  {
                  'ytick.direction': 'in', 'xtick.direction': 'in',
                  'font.sans-serif': ['CMU Sans Serif','Helectiva','Arial', 'Liberation Sans', 'Bitstream Vera Sans', 'sans-serif'],
                  }
                    )
plt.rcParams['axes.unicode_minus'] = False
colors = sns.color_palette('Set2')

# Overal graph layout and title
ncols=1
nrows=1
fig, ax = plt.subplots(nrows=nrows, ncols=ncols,figsize=(20,10))
#fig.suptitle('FF Fitting Quality',y=0.95,fontweight='bold', fontsize=14)
ax.set_xlabel(r'$\mathbf{T \ (K)}$',fontsize=36, fontweight='bold')
#ax.set_xlabel(r'T (K)',fontsize=24, fontweight='bold')
#ax.set_ylabel(r'$\mathbf{B_2}$ (cm$^3$ mol$^-1$)',fontsize=36,fontweight='bold')
ax.set_ylabel(r'$\mathbf{B_2 \ (cm^3 mol^{-1})}$',fontsize=36,fontweight='bold')


ms=20
mew=3

# Experimental data
x = exp1_data['T']
y = exp1_data['B']
f = interp1d(x, y, kind='cubic')
xnew = np.arange(np.min(x),np.max(x),0.01)
ax.plot(xnew,f(xnew),
#ax.plot(x,y,
        '-',
        color='k',
        label='Duska et al.',
        linewidth=1.4*mew,
        zorder=10
        )

# Experimental data, EOS
## x = mastiff_data['T']
## #f = interp1d(x, y, kind='cubic')
## xnew = np.arange(np.min(x),np.max(x),0.01)
## ynew = experimental_eos(xnew)
## #ax.plot(xnew,f(xnew),
## ax.plot(xnew,ynew,
##         color='k',
##         label='Experiment',
##         linewidth=1.4*mew,
##         zorder=10
##         )
x_min = np.min(x)
x_max = np.max(x)
y_min = np.min(y)
y_max = np.max(y)

# ISA data
x = mastiff_data['T']
y = mastiff_data['B']
ax.plot(x,y,
        'o',
        ms=ms,
        markeredgecolor=colors[0],
        markeredgewidth=mew,
        label='MASTIFF',
        alpha=0.75,
        zorder=5,
        )
x_min = min(x_min,np.min(x))
x_max = np.max(x)
#x_max = max(x_max,np.max(x))
# y_min = min(y_min,np.min(y))
y_max = max(y_max,np.max(y))


# Experimental data
if plot_exp2:
    x = exp2_data['T']
    y = exp2_data['B']
    #f = interp1d(x, y, kind='cubic')
    xnew = np.arange(np.min(x),np.max(x),0.01)
    #ax.plot(xnew,f(xnew),
    ax.plot(x,y,
            's',
            color='k',
            label='MBPOL',
            linewidth=1.4*mew,
            zorder=10
            )

# This paper Slater-ISA FF data
x = isaff_data['T']
y = isaff_data['B']
ax.plot(x,y,
        's',
        ms=ms,
        fillstyle='left',
        markeredgecolor=colors[1],
        markeredgewidth=mew,
        label='Aniso-Iso FF',
        alpha=0.75,
        zorder=4,
        )
x_min = min(x_min,np.min(x))
x_max = max(x_max,np.max(x))
y_min = min(y_min,np.min(y))
y_max = max(y_max,np.max(y))

# This paper Iso-Iso FF data
x = isoff_data['T']
y = isoff_data['B']
ax.plot(x,y,
        'd',
        ms=ms,
        fillstyle='none',
        markeredgecolor=colors[2],
        markeredgewidth=mew,
        label='Iso-Iso FF',
        alpha=0.75,
        zorder=3,
        )
## x_min = min(x_min,np.min(x[2:]))
## x_max = max(x_max,np.max(x[2:]))
## y_min = min(y_min,np.min(y[2:]))
## y_max = max(y_max,np.max(y[2:]))

## # JPC 2013 SAPT-FF data
## x = saptff2_data['T']
## y = saptff2_data['B']
## ax.plot(x,y,
##         's',
##         fillstyle='none',
##         markeredgecolor=colors[1],
##         markeredgewidth=mew,
##         color=colors[1],
##         ms=ms,
##         label='Born-Mayer-IP FF (prior work)',
##         )
## x_min = min(x_min,np.min(x))
## x_max = max(x_max,np.max(x))
## y_min = min(y_min,np.min(y))
## y_max = max(y_max,np.max(y))
dimer = 'h2o.png'
arr_fig = read_png(dimer)
imagebox = OffsetImage(arr_fig, zoom=0.60)
xy = [0.1, 0.1]
xybox = [0.50,0.35]
ab = AnnotationBbox(imagebox, xy,
        xybox=xybox,
        xycoords='axes fraction',
        boxcoords="axes fraction",
        frameon=False,
                )
#arrowprops=dict(arrowstyle="->"))
mol = ax.add_artist(ab)


# Axes scaling and title settings
scale = 0.02
## x_min = np.amin(x)
## x_max = np.amax(x)
# x_min = ax.get_xlim()[0]
# x_max = ax.get_xlim()[1]
xlims = [ x_min - scale*abs(x_max - x_min), 
         x_max + scale*abs(x_max - x_min) ]
## y_min = np.amin(y)
## y_max = np.amax(y)
# y_min = ax.get_ylim()[0]
# y_max = ax.get_ylim()[1]
ylims = [ y_min - scale*abs(y_max - y_min), 
         y_max + scale*abs(y_max - y_min) ]
ax.set_xlim(xlims)
ax.set_ylim(ylims)

# Legend
ax.legend(loc=8,
        #title='Method',
        fontsize=32,
        bbox_to_anchor=(0.75,0.23),
        bbox_transform=ax.figure.transFigure)
ax.get_legend().get_title().set_fontsize(16)
ax.get_legend().get_title().set_fontweight('bold')

sns.despine()

fig.savefig('h2o_2nd_virial.pdf',bbox_inches='tight',dpi=200)
#fig.savefig('h2o_2nd_virial.png',bbox_inches='tight')
#plt.show()

###########################################################################
###########################################################################
