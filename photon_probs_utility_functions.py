import matplotlib
import matplotlib.pyplot as plt
import os
import numpy as np
import scipy
from scipy.optimize import curve_fit
from scipy.special import factorial
from scipy.stats import poisson
from scipy.stats import betaprime
from scipy.special import gammainc
from scipy.special import gammaincc
from scipy.signal import convolve
from decimal import Decimal
import numpy as np
import math
import glob
import time

def gen_PN(N, Imin, Imax, alpha):
    # If one molecule is in the confocal volume, this generates the distribution of photon emissions expected.
    # N is a numpy array of the photon emission values we are interested in evaluating.
    PN = np.divide((gammaincc(N, alpha*Imin) - gammaincc(N, alpha*Imax)), N)
    if len(np.where(N==0)[0])!=0:
        for zero in np.where(N==0)[0]:
            PN[zero] = scipy.special.expi(-alpha*Imax)-scipy.special.expi(-alpha*Imin)
    PN = np.nan_to_num(PN)
    if np.sum(PN) != 0:
        PN /= np.sum(PN)
    return PN



def gen_PS(S,PSoriginal,PN,normalise=True):
    # Given probabilities (PSoriginal) of n-1 molecules giving rise to numbers of photons from 1 up to n*Nmax, what are the probabilities of
    # n molecules giving rise to sum S, where S is a numpy array or list of values.
    # Speed up by somehow ignoring all the zero values?
    PS = []
    Smax = np.amax(S)
    PN = np.pad(PN, (0, int(Smax + 1 - len(PN))), 'constant', constant_values=(0))
    PSoriginal = np.pad(PSoriginal,(0,int(Smax+1-len(PSoriginal))),'constant',constant_values=(0))
    for s in S:
        PS.append(np.sum(np.multiply(PN[:s+1],np.flip(PSoriginal[:s+1]))))
    if normalise:
        PS /= np.sum(PS)
    return PS

def gen_PM(max_mols,c0,flowrate,timebin,sigmax,sigmay,Imin,Imax):
    # Distribution of the number of molecules that get the chance to emit (i.e. everything that passes through the
    # ellipse where laser intensity >= Imin): unitless (just number).
    # max_mols is the maximum number of molecules that we want to consider. The output will be an array of length max_mols+1
    # of the probabilities of number of molecules = 0 up to no. of molecules = max_mols.
    # c0 is the concentration of molecules: nanomolar. The 6.02e17 is to convert from nanomolar to number of molecules per m^3
    # flowrate is the flow rate in units of metres per second (ie z direction flow rate)
    # timebin is the length of the timebin
    # sigmax and sigmay are the laser intensity standard deviations in the x and y directions, respectively.
    # Imin is the minimum laser intensity that we consider i.e. the boundary of the ellipse.
    # Imax is the maximum laser intensity - we set this to be 1.
    # flowrate for FFE is approx. 4 mm/s so if flowrate is 0.004, then if sigmax and sigmay are in metres, conc is in molecules/m3
    n0 = c0*(6.02e17)*flowrate*timebin*2*math.pi*sigmax*sigmay*np.log(Imax/Imin)
    mols = np.arange(max_mols+1)
    PM = poisson.pmf(mols,n0)
    PM = np.nan_to_num(PM)
    return PM

def gen_PD(alpha,max_mols,c0,flowrate,timebin,sigmax,sigmay,Imin,Imax,PS_directory=None, save_all_PS = False, maxphotons = 14500, norm=True, PSnorm=True):
    # Add on 3*standard deviations of poisson deviation
    # The numpy amax of that and 6 is incase the left side is smaller than 1 which would result in weird artefacts that the molecule
    # will never emit any photon. We have 6 to make sure we have a reasonable range of photon emissions to consider (6 itself is
    # a fairly arbitrary value though).
    alpha = np.abs(alpha)
    c0 = np.abs(c0)
    all_PS_loaded = False
    if PS_directory:
        if not os.path.exists(PS_directory):
            os.mkdir(PS_directory)
        PS_directory += '/alpha{}_maxmols{}_maxphotons{}'.format(alpha, max_mols, maxphotons)
        if not os.path.exists(PS_directory):
            os.mkdir(PS_directory)
        PS_directory += '/'
        if os.path.exists(PS_directory+'all_PS.npy'):
            all_PS = np.load(PS_directory+'all_PS.npy')
            all_PS_loaded = True
    # Get prob distribution of numbers of molecules in confocal volume
    PM = gen_PM(max_mols, c0, flowrate, timebin, sigmax, sigmay, Imin, Imax)

    if not all_PS_loaded:
        N = np.arange(np.amax([int(alpha*Imax+ 3*np.sqrt(alpha*Imax)),6]))
        PN = gen_PN(N, Imin, Imax, alpha)[:maxphotons+1]

        P0 = np.zeros_like(PN)
        P0[0] = 1

        all_PS = np.vstack((P0,PN))

        PScurrent = np.copy(PN)
        for i in range(2, max_mols+1):
            if PS_directory and save_all_PS:
                PScurrent_filename = '{}PS{}.npy'.format(PS_directory,i)
                if os.path.exists(PScurrent_filename):
                    PScurrent = np.load(PScurrent_filename)
                else:
                    PScurrent = combine_dists(np.arange(len(PN)), PN, np.arange(len(PScurrent)), PScurrent)
                    np.save(PScurrent_filename, PScurrent)
            else:
                PScurrent = combine_dists(np.arange(len(PN)), PN, np.arange(len(PScurrent)), PScurrent)[:maxphotons+1]

            all_PS = np.pad(all_PS,((0,0),(0,len(PScurrent)-len(all_PS[0]))))

            all_PS = np.vstack((all_PS, PScurrent))

        if PS_directory:
            np.save('{}all_PS.npy'.format(PS_directory), all_PS)

    all_PS_withconcs = all_PS * PM[:, np.newaxis]
    probs_withconcs = np.sum(all_PS_withconcs, axis=0)
    if norm:
        probs_withconcs /= np.sum(probs_withconcs)
    return probs_withconcs


def get_trace_likelihood(exp_timetrace_y,model_x,model_y):
    model_y = np.pad(model_y, (int(model_x[0]), int(np.amax([int(np.amax(exp_timetrace_y))-model_x[-1],0]))), 'constant', constant_values=(0))
    model_y = np.array(model_y)
    likelihood = 0
    for i,photoncounts in enumerate(exp_timetrace_y):
        likelihood += np.log(model_y[int(photoncounts)])
    return -likelihood

def get_likelihood_bg_betaprime(to_vary,exp_timetrace_x,bg_timetrace_y):
    x = np.arange(np.amax(exp_timetrace_x)+1)
    a,b = to_vary
    return get_trace_likelihood(bg_timetrace_y,x,betaprime.pdf(x, a, b)+1e-20)

def fit_bg_betaprime(exp_timetrace_x,bg_timetrace_y,savefig=None):
    a = 0.5
    b = 1
    result = scipy.optimize.minimize(get_likelihood_bg_betaprime,np.array([a,b]),(exp_timetrace_x,bg_timetrace_y),method='Nelder-Mead')
    fitted_bg = betaprime.pdf(exp_timetrace_x,result.x[0],result.x[1])
    fitted_bg[0] = fitted_bg[-1]
    if savefig:
        fig, ax = plt.subplots()
        ax.plot(exp_timetrace_x,fitted_bg)
        ax.hist(bg_timetrace_y, bins=np.arange(500), density=True)
        fig.savefig(savefig+'.svg')
        ax.set_yscale('log')
        fig.savefig(savefig+'_log.svg')
    return fitted_bg


def combine_dists(dist1x,dist1y,dist2x,dist2y):
    # Combine the distributions 1 and 2, where x is the number of photons emitted and y is the probability of photon counts.
    if len(dist2x)>0:
        # Cut down both distributions to the non zero probability entries
        where_nonzero_1 = np.argwhere(dist1y)
        if np.size(where_nonzero_1) != 0:
            new_dist1y = dist1y[:where_nonzero_1[-1][0]+1]
        else:
            new_dist1y = dist1y

        where_nonzero_2 = np.argwhere(dist2y)
        if np.size(where_nonzero_2) > 0:
            new_dist2y = dist2y[:where_nonzero_2[-1][0] + 1]
        else:
            new_dist2y = dist2y

        total_probs = convolve(new_dist1y, new_dist2y, method='direct')

    else:
        total_probs = dist1y
    if len(total_probs) < len(dist1y)+len(dist2y):
        total_probs = np.pad(total_probs, (0, len(dist1y)+len(dist2y)-len(total_probs)))
    return total_probs

def gen_PD_withnoise(alpha,max_mols,c0,flowrate,timebin,sigmax,sigmay,Imin,Imax,noisedist_x,noisedist_y,PS_directory=None):
    model_dist_y = gen_PD(alpha,max_mols,c0,flowrate,timebin,sigmax,sigmay,Imin,Imax,PS_directory=PS_directory)
    model_dist_x = np.arange(len(model_dist_y))
    return combine_dists(model_dist_x,model_dist_y,noisedist_x,noisedist_y)

def change_timebins(timetrace_x, timetrace_y, original_timebins, new_timebins):
    if (new_timebins/original_timebins)%1 != 0:
        print('Error in changing timebins. Please ensure new timebins is a multiple of original timebins. Terminating.')
        exit()
    if original_timebins != new_timebins:
        timebins_to_combine = int(new_timebins/original_timebins)
        to_cut = len(timetrace_y) % timebins_to_combine
        if to_cut != 0:
            timetrace_y = np.reshape(timetrace_y[:-to_cut],
                                         (int(len(timetrace_y[:-to_cut]) / timebins_to_combine),
                                          timebins_to_combine))
            timetrace_y = np.squeeze(np.sum(timetrace_y, axis=1))
            timetrace_x = timetrace_x[0:-to_cut:timebins_to_combine]
        else:
            timetrace_y = np.reshape(timetrace_y,
                                         (int(len(timetrace_y) / timebins_to_combine), timebins_to_combine))
            timetrace_y = np.squeeze(np.sum(timetrace_y, axis=1))
            timetrace_x = timetrace_x[0:len(timetrace_x):timebins_to_combine]
    return np.array([timetrace_x, timetrace_y]).T

def round_precision(x, precision):
    lower = np.floor(x/precision)
    upper = np.ceil(x/precision)
    if lower == upper:
        return x
    else:
        remainder = x - lower*precision
        if remainder < precision/2:
            return x-remainder
        else:
            return x+precision-remainder




def get_distribution(alphas, concs, max_mols, flowrate, timebin, sigmax, sigmay, Imin, Imax,
                     bg_x=None, bg_y=None, PS_directory=None, max_photons = None):
    individual_dists = []
    for i,alpha in enumerate(alphas):
        dist = gen_PD(alpha, max_mols[i], concs[i], flowrate, timebin, sigmax, sigmay, Imin, Imax, PS_directory=PS_directory)
        if not max_photons:
            individual_dists.append(dist)
        else:
            individual_dists.append(dist[:max_photons+1])
    dist = individual_dists[0]
    for i in range(1,len(alphas)):
        dist = combine_dists(np.arange(len(dist)), dist, np.arange(len(individual_dists[i])), individual_dists[i])
        if max_photons:
            dist = dist[:max_photons+1]
    if bg_x is not None and bg_y is not None:
        dist = combine_dists(np.arange(len(dist)), dist, bg_x, bg_y)

    dist = np.nan_to_num(np.abs(dist))

    if 0 in dist:
        if np.sum(dist) == 0:
            return np.full_like(dist, 1e-300)
        dist[dist == 0] = np.amin(dist)
    if max_photons:
        dist = dist[:max_photons+1]
    return dist

def dist_betaprime(dist_params, oligo_sizes):
    # dist_params is a list of the parameters which define the distribution
    # oligo_sizes is an array from 2 to the max nmer size allowed
    # The function returns an array of the concentrations of all the oligomer species (nM)
    # a, b, c = [10**param for param in dist_params]
    a,b,c = dist_params
    result = c*betaprime.pdf(oligo_sizes, a, b)

    return result

def dist_betaprime_reparam(dist_params, oligo_sizes):
    # dist_params is a list of the parameters which define the distribution
    # oligo_sizes is an array from 2 to the max nmer size allowed
    # The function returns an array of the concentrations of all the oligomer species (nM)
    mu, v, c = dist_params
    b = v + 2
    a = mu * (b+1)
    result = c * betaprime.pdf(oligo_sizes, a, b)
    return result

def dist_unnorm_betaprime(dist_params, oligo_sizes):
    # dist_params is a list of the parameters which define the distribution
    # oligo_sizes is an array from 2 to the max nmer size allowed
    # The function returns an array of the concentrations of all the oligomer species
    a, b, c = dist_params

    return c * betaprime.pdf(oligo_sizes, a, b)


def string_to_list(stringlist, type='float'):
    # stringlist is a string e.g. '[1,2,3]' that must be formatted as though it were a list within python.
    if '[[' in stringlist:
        stringlist = stringlist[2:-2]
        stringlist = stringlist.split('],[')
        to_return = []
        for sublist in stringlist:
            sublist = [float(i) for i in sublist.split(',')]
            to_return.append(sublist)
        return(to_return)

    if '[' not in stringlist or ']' not in stringlist:
        print('Error in config file: please format list items with []. Terminating.')
        exit()
    if ',' in stringlist:
        if type == 'float':
            return [float(i) for i in stringlist.strip('"').strip('[').strip(']').split(',')]
        elif type == 'string':
            return [i.strip(' ') for i in stringlist.strip('"').strip('[').strip(']').split(',')]
        elif type == 'int':
            return [int(i) for i in stringlist.strip('"').strip('[').strip(']').split(',')]
    else:
        try:
            return [float(stringlist.strip('[').strip(']'))]
        except ValueError:
            return 1


def combine_traces(traces_to_combine_filenames):
    traces_to_combine = []
    for file in traces_to_combine_filenames:
        trace = np.genfromtxt(file)
        traces_to_combine.append(trace)
    print(traces_to_combine)




def model_likelihood_fitalpha(to_vary,max_mols,c0,flowrate,timebin,sigmax,sigmay,Imin,Imax,noisedist_x,noisedist_y,
                              exp_timetrace_y,save_base,PS_directory=None):
    alpha = np.abs(to_vary[0])
    model_y = gen_PD_withnoise(alpha,max_mols,c0,flowrate,timebin,sigmax,sigmay,Imin,Imax,noisedist_x,noisedist_y,PS_directory=PS_directory)
    np.save(save_base+'model_y', model_y)
    likelihood = get_trace_likelihood(exp_timetrace_y, np.arange(len(model_y)), model_y)
    if not os.path.exists(save_base+'likelihoods_fitalpha.npy'):
        likelihoods = np.array([alpha,c0,likelihood],dtype=object)
    else:
        likelihoods = np.load(save_base+'likelihoods_fitalpha.npy',allow_pickle=True)
        likelihoods = np.vstack((likelihoods,np.array([alpha,c0,likelihood])))
    np.save(save_base+'likelihoods_fitalpha.npy', likelihoods)
    return likelihood

def model_likelihood_fitconc(to_vary,alpha,max_mols,flowrate,timebin,sigmax,sigmay,Imin,Imax,noisedist_x,noisedist_y,
                              exp_timetrace_y,save_base,PS_directory):
    c0 = np.abs(to_vary[0])
    model_y = gen_PD_withnoise(alpha,max_mols,c0,flowrate,timebin,sigmax,sigmay,Imin,Imax,noisedist_x,noisedist_y,PS_directory=PS_directory)
    np.save(save_base + 'model_y', model_y)
    likelihood = get_trace_likelihood(exp_timetrace_y, np.arange(len(model_y)), model_y)
    if not os.path.exists(save_base+'likelihoods_fitconc.npy'):
        likelihoods = np.array([alpha,c0,likelihood],dtype=object)
    else:
        likelihoods = np.load(save_base+'likelihoods_fitconc.npy',allow_pickle=True)
        likelihoods = np.vstack((likelihoods,np.array([alpha,c0,likelihood])))
    np.save(save_base+'likelihoods_fitconc.npy', likelihoods)
    print('fit', c0, likelihood)

    exp_bins = np.arange(-0.5, np.amax(exp_timetrace_y), 1)
    exp_entries, exp_bin_edges = np.histogram(exp_timetrace_y, bins=exp_bins, density=True)
    exp_bin_middles = 0.5 * (exp_bin_edges[1:] + exp_bin_edges[:-1])
    fig, ax = plt.subplots()
    ax.plot(exp_bin_middles, exp_entries, color='xkcd:cobalt blue', label='experimental')
    ax.plot(np.arange(len(model_y)), model_y, color='xkcd:azure', label='fit')
    ax.set_xlabel('Photon counts per timebin')
    ax.set_ylabel('Probability')
    plt.legend()
    ax.text(0.1, 0.9, 'Likelihood: {0:.0f}'.format(likelihood), transform=ax.transAxes)
    if len(np.shape(likelihoods)) > 1:
        ax.set_title(' Conc: {0:.3f}'.format(likelihoods[-1][1]))
    fig.savefig(save_base + 'fit.svg')
    ax.set_xlim((0, int(np.where(exp_entries != 0)[0][-1] + 10)))
    fig.savefig(save_base + 'fit_cutx.svg')
    plt.close()
    return likelihood

def model_likelihood_fitalpha_fitconc(to_vary,max_mols,flowrate,timebin,sigmax,sigmay,Imin,Imax,noisedist_x,noisedist_y,
                              exp_timetrace_y,save_base,PS_directory=None):
    alpha,c0 = np.abs(to_vary)

    model_y = gen_PD_withnoise(alpha,max_mols,c0,flowrate,timebin,sigmax,sigmay,Imin,Imax,noisedist_x,noisedist_y,PS_directory=PS_directory)

    np.save(save_base + 'model_y', model_y)
    likelihood = get_trace_likelihood(exp_timetrace_y, np.arange(len(model_y)), model_y)
    if not os.path.exists(save_base+'likelihoods_fitalpha_fitconc.npy'):
        likelihoods = np.array([alpha,c0,likelihood],dtype=object)
    else:
        likelihoods = np.load(save_base+'likelihoods_fitalpha_fitconc.npy',allow_pickle=True)
        likelihoods = np.vstack((likelihoods,np.array([alpha,c0,likelihood])))
    np.save(save_base+'likelihoods_fitalpha_fitconc.npy', likelihoods)
    print('fit',alpha,c0,likelihood)

    exp_bins = np.arange(-0.5, np.amax(exp_timetrace_y), 1)
    exp_entries, exp_bin_edges = np.histogram(exp_timetrace_y, bins=exp_bins, density=True)
    exp_bin_middles = 0.5 * (exp_bin_edges[1:] + exp_bin_edges[:-1])
    fig, ax = plt.subplots()
    ax.plot(exp_bin_middles, exp_entries, color='xkcd:cobalt blue',label='experimental')
    ax.plot(np.arange(len(model_y)), model_y, color='xkcd:azure',label='fit')
    ax.set_xlabel('Photon counts per timebin')
    ax.set_ylabel('Probability')

    plt.legend()
    ax.text(0.1, 0.9, 'Likelihood: {0:.0f}'.format(likelihood), transform=ax.transAxes)
    if len(np.shape(likelihoods))>1:
        ax.set_title('Alpha: {0:.1f}'.format(likelihoods[-1][0]) + ' Conc: {0:.3f}'.format(likelihoods[-1][1]))
    fig.savefig(save_base + 'fit.svg')
    ax.set_xlim((0, int(np.where(exp_entries != 0)[0][-1] + 10)))
    fig.savefig(save_base + 'fit_cutx.svg')
    plt.close()
    return likelihood
