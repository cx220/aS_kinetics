import matplotlib
import matplotlib.pyplot as plt
import os
import numpy as np
import scipy
from scipy.optimize import curve_fit
from scipy.special import factorial
from scipy.stats import poisson
from scipy.special import gammainc
from scipy.special import gammaincc
import matplotlib.patches as mpatches

import numpy as np
import math
import glob

from numba import jit
import readPTU as rp

import photon_probs_utility_functions

# PARAMETERS THAT MAY BE CHANGED FOR FITTING
save_dir = 'save_dir'
fit_dir = 'fit_dir'
PS_dir = 'PS_dir'

if not os.path.exists(save_dir):
    os.mkdir(save_dir)

fit_dir = save_dir+'/'+fit_dir
if not os.path.exists(fit_dir):
    os.mkdir(fit_dir)
fit_dir+='/'

summary_file = open(fit_dir+"summary.txt", "w")
summary_file.write('PS_dir: '+PS_dir)

# Timetrace data: numpy array or csv
exp_data = example.csv

summary_file.write('\nfile to analyse: '+exp_data)
if exp_data[-3:] == 'npy':
    exp_data = np.load(exp_data,allow_pickle=True)
if exp_data[-3:] == 'csv':
    exp_data = np.loadtxt(exp_data, delimiter=',')

exp_timetrace_x = exp_data[:,0]
exp_timetrace_y = exp_data[:,1]

fig,ax = plt.subplots()
ax.plot(exp_data[:,0],exp_data[:,1],label='original')
ax.set_xlim((2,100))
fig.savefig(fit_dir+'timetracesection.png')background_file = None


# Straight channel
max_mols = 450
flowrate = 5.5e-3
sigmax = 1e-6
sigmay = 1e-6
Imin = 0.001
Imax = 1

# Number of species to be considered
number_of_species = 1
summary_file.write('\nnumber_of_species: ' + str(number_of_species))
fit_alpha = True
initial_alpha = 22
fit_conc = True
initial_conc = 0.5

if number_of_species == 1:
    summary_file.write('\nfit_alpha: '+str(fit_alpha))
    if fit_alpha:
        summary_file.write('\ninitial_alpha: ' + str(initial_alpha))
    else:
        alpha = 22
        summary_file.write('\nalpha: ' + str(alpha))
    summary_file.write('\nfit_conc: ' + str(fit_conc))
    if fit_conc:
        summary_file.write('\ninitial_conc: ' + str(initial_conc))
    else:
        conc = 0.1
        summary_file.write('\nconc: ' + str(conc))

    summary_file.write(
        '\nflowrate: ' + str(flowrate) + '\nsigmax: ' + str(sigmax) + '\nsigmay: ' + str(sigmay) + '\nImin: ' + str(
            Imin) + '\nImax: ' + str(Imax) + '\nmax_mols: ' + str(max_mols))

if background_file:
    with rp.PTUfile(background_file) as ptu_file:
        ptu_meas = rp.PTUmeasurement(ptu_file)
    bg_timetrace_x, bg_timetrace_y, bg_timetrace_recnum = ptu_meas.timetrace(resolution=timetrace_resolution, n_threads=4,
                                                                          record_range=None)
else:
    bg_timetrace_x, bg_timetrace_y = np.arange(10),np.zeros_like(np.arange(10))

background_file = None

if background_file:
    summary_file.write('\nbackground file: '+ background_file)
else:
    summary_file.write('\nbackground file: None')


bg_csv = True
if background_file:
    bg_data = np.loadtxt(background_file, delimiter=',')

    bg_timetrace_x = bg_data[:,0]
    bg_timetrace_y = bg_data[:,1]
    original_timebins = 1e-3
    new_timebins = 2e-3
    timebins_to_combine = int(new_timebins/original_timebins)
    to_cut = len(bg_timetrace_y)%timebins_to_combine
    if to_cut > 0:
        bg_timetrace_y = np.reshape(bg_timetrace_y[:-to_cut],(int(len(bg_timetrace_y[:-to_cut])/timebins_to_combine),timebins_to_combine))
        bg_timetrace_x = bg_timetrace_x[0:-to_cut:timebins_to_combine]
    else:
        bg_timetrace_y = np.reshape(bg_timetrace_y,(int(len(bg_timetrace_y) / timebins_to_combine), timebins_to_combine))
        bg_timetrace_x = bg_timetrace_x[0::timebins_to_combine]
    bg_timetrace_y = np.squeeze(np.sum(bg_timetrace_y,axis=1))
else:
    bg_timetrace_x, bg_timetrace_y = np.arange(10), np.zeros_like(np.arange(10))

bg_bins = np.arange(-0.5, np.amax(bg_timetrace_y), 1)
bg_entries, bg_bin_edges = np.histogram(bg_timetrace_y, bins=bg_bins, density=True)
bg_bin_middles = 0.5 * (bg_bin_edges[1:] + bg_bin_edges[:-1])

exp_bins = np.arange(-0.5, np.amax(exp_timetrace_y), 1)
exp_entries, exp_bin_edges = np.histogram(exp_timetrace_y, bins=exp_bins, density=True)
exp_bin_middles = 0.5 * (exp_bin_edges[1:] + exp_bin_edges[:-1])

fig,ax = plt.subplots()
ax.plot(exp_timetrace_x,exp_timetrace_y,color='xkcd:cobalt blue')
ax.set_xlabel('Time /s')
ax.set_ylabel('Photons per timebin')
fig.savefig(fit_dir+'timetrace.svg')


if number_of_species == 1:
    if fit_alpha and not fit_conc:
        result = scipy.optimize.minimize(photon_probs_utility_functions.model_likelihood_fitalpha,np.array([initial_alpha]),
                                (max_mols,conc,flowrate,timetrace_resolution,sigmax,sigmay,Imin,Imax,PS_dir,bg_bin_middles,
                                 bg_entries,exp_timetrace_y,fit_dir),method='Nelder-Mead',options={'xatol':0.1})
        likelihoods = np.load(fit_dir+'likelihoods_fitalpha.npy',allow_pickle=True)

    elif not fit_alpha and fit_conc:
        result = scipy.optimize.minimize(photon_probs_utility_functions.model_likelihood_fitconc,np.array([initial_conc]),
                                (alpha,max_mols,flowrate,timetrace_resolution,sigmax,sigmay,Imin,Imax,PS_dir,bg_bin_middles,
                                 bg_entries,exp_timetrace_y,fit_dir),method='Nelder-Mead',options={'xatol':0.1})
        likelihoods = np.load(fit_dir+'likelihoods_fitconc.npy',allow_pickle=True)

    elif fit_alpha and fit_conc:
        result = scipy.optimize.minimize(photon_probs_utility_functions.model_likelihood_fitalpha_fitconc,np.array([initial_alpha,initial_conc]),
                                (max_mols,flowrate,timetrace_resolution,sigmax,sigmay,Imin,Imax,bg_bin_middles,
                                 bg_entries,exp_timetrace_y,fit_dir),method='Nelder-Mead',options={'xtol':0.1})
        likelihoods = np.load(fit_dir+'likelihoods_fitalpha_fitconc.npy',allow_pickle=True)


summary_file.write('\nfit result: '+str(likelihoods[-1][:-1]))
summary_file.write('\nlikelihood: '+str(likelihoods[-1][-1]))

