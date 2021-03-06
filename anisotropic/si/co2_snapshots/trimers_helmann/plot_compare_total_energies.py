#!/usr/bin/env python
"""

Last Updated:
"""

# Standard modules
import argparse
import numpy as np
import sys
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib.patches as patches
import itertools
from matplotlib._png import read_png
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from mpl_toolkits.axes_grid1 import ImageGrid
# from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from matplotlib.patches import ConnectionPatch
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
parser = argparse.ArgumentParser()
displayhelp="In addition to saving the file, immediately display the plot using plt.show()"
fadehelp="Weight the opacity of each point depending on the SAPT Total Energy."
asymptotichelp="""Only display the asymptotic_scale fraction of points with the
lowest SAPT exchange energies (which serve as a proxy for the most asymptotic
points)"""
asymptoticscalehelp="Dictates the fraction of points that will be displayed.  See asymptotic above."

#parser.add_argument("energy_file", type=str, help=energyhelp)
parser.add_argument("-p","--prefixes", help="prefixes", nargs="*", default=['mastiff_3b_'])
parser.add_argument("-s","--suffixes", help="suffixes", nargs="*", default=['.dat'])

parser.add_argument("--display", help=displayhelp,\
         action="store_true", default=False)

parser.add_argument("--asymptotic", help=asymptotichelp,\
         action="store_true", default=False)
parser.add_argument("--asymptotic_scale", help=asymptoticscalehelp,\
         type=float,default=0.05)

parser.add_argument("--fade", help=fadehelp,\
         action="store_true", default=False)

parser.add_argument("-m","--molecule",help=fadehelp,)

args = parser.parse_args()

component_prefixes = args.prefixes
component_suffixes = args.suffixes

## component_prefixes = ['isa_constrained_exp_','saptff_constrained_exp_']
## component_suffixes = ['_unconstrained.dat','_unconstrained.dat']

## if component_suffixes == []:
##     component_prefixes = ['fit_exp_']
##     component_suffixes = ['_unconstrained.dat']


###########################################################################
###########################################################################


###########################################################################
########################## Main Code ######################################

# Read data from each energy component
# Set some global color preferences for how the graph's colors should look
## sns.set_context('paper')
## sns.axes_style("darkgrid")
## sns.set_style("ticks")
## sns.set_color_codes()
## #palette = itertools.cycle(sns.color_palette())
## #palette = sns.color_palette()
plt.rcParams['axes.unicode_minus'] = False
sns.set(context='talk',font_scale=1.1)
palette = sns.color_palette("Set2")
# Reorder palette
#palette = palette[2:3] + palette[1:2] + palette[0:]

sns.set_style("ticks", 
                  {
                  'ytick.direction': 'in', 'xtick.direction': 'in',
                  'font.sans-serif': ['CMU Sans Serif','Helectiva','Arial', 'Liberation Sans', 'Bitstream Vera Sans', 'sans-serif'],
                  }
                    )

# Overal graph layout and title
ncols=len(args.prefixes)
nrows=2
fig, axes = plt.subplots(nrows=nrows, ncols=ncols,
                    figsize=(5*ncols+2,10),
                    # sharex=True,
                    # sharey=True
                    )
fig.text(0.5,0.035, 'CCSD(T) Energy (kJ/mol)',ha='center',
        va='center',
        fontweight='semibold',
        fontsize=24)
fig.text(0.100,0.5, 'MASTIFF-CC Energy (kJ/mol)',
        ha='center', va='center',
        rotation='vertical',
        fontweight='semibold',
        fontsize=24)

first=True
labels = []
label_texts = itertools.cycle(['3b Energy','Aniso-Iso FF','Aniso-Aniso FF'])
zorder = itertools.cycle([8,9,10])
m_values = []
b_values = []
for i, (component_prefix,component_suffix) in enumerate(zip(component_prefixes,component_suffixes)):
    last = (component_prefix == component_prefixes[-1])
    last = (last and component_suffix == component_suffixes[-1])

    color = palette[i]

    # Filenames to read in 
    induction_file = component_prefix +  'induction' + component_suffix
    dispersion_file = component_prefix +  'dispersion' + component_suffix
    total_2b_energy_file = 'ccsdt_2b_energy.dat'
    total_3b_energy_file = 'ccsdt_3b_energy.dat'

    induction = pd.read_csv(
                    induction_file,delim_whitespace=True,names=['ff'],skiprows=1)
    dispersion = pd.read_csv(
                    dispersion_file,delim_whitespace=True,names=['i','ff','unit'],skiprows=1)
    total_2b_energy = pd.read_csv(
                    total_2b_energy_file,delim_whitespace=True,names=['qm'],skiprows=1)
    total_3b_energy = pd.read_csv(
                    total_3b_energy_file,delim_whitespace=True,names=['foo','qm','unit'],skiprows=1)

    # Convert units from Kelvin to kJ/mol
    kelvin2kJmol = 0.00831435
    total_2b_energy['qm'] *= kelvin2kJmol
    total_3b_energy['qm'] *= kelvin2kJmol

    ff_3b = np.array(induction['ff'] + dispersion['ff'])
    d = {'ff': ff_3b, 'qm3b': np.array(total_3b_energy['qm']), 'qm2b' :
            np.array(total_2b_energy['qm']) }
    total_energy = pd.DataFrame(data=d.values(),index=d.keys())
    total_energy = total_energy.transpose()

    # Plot each energy component
    count=0
    titles=['Total Energy','Total Energy (Attractive Region)']


    eng = total_energy['qm2b']
    min_energy = np.min(eng)
    max_energy = np.max(eng)

    if args.asymptotic:
        scale = args.asymptotic_scale
    else:
        scale = 1.0
    xmax = scale*(max_energy - min_energy) + min_energy
    xmin = min_energy

    include_points = np.all([xmin*np.ones_like(eng) <= eng,
                             xmax*np.ones_like(eng) >= eng],axis=0)
    include_points = np.array(eng > 0)

    print include_points
    print np.sum(include_points)


    #for component in electrostatics,exchange,dispersion,None,induction,dhf,total_energy,total_energy:
    for ic,component in enumerate([total_energy,total_energy]):
        print axes
        print len(axes), ic, i
        count += 1
        #ax = axes[ic,i]
        ax = axes[ic]

        x = component['qm3b'][include_points]
        y = component['ff'][include_points]

        if args.fade:
            etot = total_energy['qm2b']
            alpha = np.where( etot < 0, 0.6, 0.1 )
            alpha_color = np.asarray([color + (a,) for a in alpha])

            sc = ax.scatter(x,y, color=alpha_color, 
                                s=25, lw =.75, 
                                #facecolors='none',
                                edgecolors='none',
                                zorder=10)

        else:
            sc = ax.scatter(x,y, color=color, 
                                s=25, lw =.75, 
                                #facecolors='none',
                                edgecolors='none',
                                alpha=0.6,zorder=10)

        # Axes scaling and title settings
        scale = 0.02
        xy_min = min(np.amin(x),np.amin(y))
        xy_max = max(np.amax(x),np.amax(y))
        if titles[count-1] == titles[-1]:
            xy_min *= 0.1
            xy_max *= 0.1
        m, b = np.polyfit(x, y, 1)
        m_values.append(m)
        b_values.append(b)
            ## ax.plot(np.array(lims),m*np.array(lims)+b,':',color=color)
        lims = [ xy_min - scale*abs(xy_max - xy_min), 
                 xy_max + scale*abs(xy_max - xy_min) ]
        ax.set_aspect('equal')
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        if ic == 0:
            ax.set_title(label_texts.next(),fontweight='bold',
                    #color=palette[i],
                    color='k',
                    fontsize=24)


        # Shade in region to indicate +/- 1kj/mol error in ff
        ax.plot(lims, lims, 'k-', alpha=0.5, zorder=20)
        rel = 0.1
        lims = ax.get_ylim()
        x1 = np.arange(lims[0],lims[1],0.01)
        if ic == 1:
            kjmol = 1.0
            ax.fill_between(x1,x1-kjmol,x1+kjmol,
                    edgecolor='None',
                    zorder=0,alpha=0.05,color='k')
            lims = np.array(lims)
            ax.plot(lims, lims-kjmol, 'k--', alpha=0.5, zorder=20)
            ax.plot(lims, lims+kjmol, 'k--', alpha=0.5, zorder=20)

        if ic == 1:
            #axout = axes[0,i]
            axout = axes[0]
            #axins = axes[1,i]
            axins = axes[1]
            bmin = lims[0]
            bmax= lims[1]
            print bmin, bmax
            blims = ((bmin,bmin),(bmax,bmax))
            #rec = patches.Rectangle(lims[0],xy_max-xy_min,xy_max-xy_min,
            rec = patches.Rectangle(blims[0],bmax-bmin,bmax-bmin,
                    #coords="Data",
                    fill=False,
                    ec="0.1",
                    lw=2.0,
                    zorder=30,
                    )
            axout.add_artist(rec)
            for xy in blims:
                con_patch = ConnectionPatch(
                        xyA=xy, xyB=xy, 
                        coordsA="data",
                        coordsB="data",
                        axesA=axout, axesB=axins, 
                        lw=2.0,
                        color="0.1",
                        zorder=30,
                        )
                axout.add_artist(con_patch)

        # Add RMS Error legends to each plot
        ## textstr = 'RMS Error:\n{:6.3f} kJ/mol'
        ## if ic == 0:
        ##     rms_error = np.sqrt(np.sum((x-y)**2)/len(x))
        ## else:
        ##     x1 = total_energy['qm3b'][include_points]
        ##     rms_error = np.sqrt(np.sum(
        ##                                 np.select([x1 < 0], [(x-y)**2])/len(x1[x1<0])))
        ##     textstr = 'a' + textstr
        textstr = 'RMS Error:\n{:6.3f} kJ/mol'
        if ic == 0:
            rms_error = np.sqrt(np.sum((x-y)**2)/len(x))
        
            textstr = textstr.format(rms_error)
            props = dict(facecolor=sns.axes_style()['axes.facecolor'], lw=0, alpha=1)
            #ax.text(0.98, 0.02, textstr, transform=ax.transAxes, 
                            # verticalalignment='bottom', horizontalalignment='right',bbox=props)
            ax.text(0.07, 0.95, textstr, transform=ax.transAxes, 
                            fontsize=16, color=palette[i],
                            verticalalignment='top', horizontalalignment='left',bbox=props)

        ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(7))
        ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(7))

        first = False

        if i == 2 and ic == 1:
            # Molecule Inset
            dimer = '{0}.png'.format(args.molecule)
            arr_fig = read_png(dimer)
            imagebox = OffsetImage(arr_fig, zoom=0.25)
            xy = [0.1, 0.1]
            xybox = [0.68,0.30]
            ab = AnnotationBbox(imagebox, xy,
                    xybox=xybox,
                    xycoords='axes fraction',
                    boxcoords="axes fraction",
                    frameon=False,
                            )
            #arrowprops=dict(arrowstyle="->"))
            mol = ax.add_artist(ab)

if args.asymptotic:
    outtitle='asymptotic_sapt_comparison.pdf'
else:
    outtitle = 'co2_trimer_comparison.pdf'
    #outtitle='sapt_comparison.pdf'
fig.savefig(outtitle,bbox_inches='tight',dpi=200)
if args.display:
    plt.draw()
    plt.show()

###########################################################################
###########################################################################
