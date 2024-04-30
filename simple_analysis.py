import matplotlib
import matplotlib.pyplot as plt
import os
import sys
import numpy as np
import scipy
import sys
from scipy.optimize import curve_fit
import numpy as np
import math
import glob
import io





def gaussian(xvals, mean, stdev, scale):
    return scale * scipy.stats.norm.pdf(xvals, mean, stdev)

def bg_correct_gaussian(trace, save_dir):
    bins = np.arange(-0.5, np.amax(trace), 1)
    entries, bin_edges = np.histogram(trace, bins=bins, density=True)
    bin_middles = 0.5 * (bin_edges[1:] + bin_edges[:-1])

    # Find the approximate bg level: fit a gaussian and take the mean to be the bg photon count
    mean_guess = bin_middles[entries == np.amax(entries)]
    popt, pcov = scipy.optimize.curve_fit(gaussian, bin_middles, entries, p0=[mean_guess[0], mean_guess[0] / 4, 1])

    fit = gaussian(bin_middles, *popt)
    bg_photons = popt[0]

    fig,ax = plt.subplots()
    ax.plot(bin_middles, entries)
    ax.plot(bin_middles, fit)
    fig.savefig('{}/bg_hist'.format(save_dir), bbox_inches='tight')
    plt.close()

    return trace, bg_photons*np.ones_like(trace)

def rolling_median(trace, window):
    return np.array([np.median(trace[i:i+window]) for i in range(len(trace)-window)])

def bg_correct_median(trace, window):
    return trace[:-window], rolling_median(trace, window)


def get_oligomers(trace_filename, bg_function, save_dir, oligomer_cutoffs, dilution_factor=1,
                              bg_extra_args = None, load_analysis = True, as_frac_median=True):
    # oligomer_cutoffs is a list of the oligomer cutoff values to be calculated ie the
    # minimum number of photons/timebin to be considered an oligomer
    if load_analysis and os.path.exists('{}/oligomers.txt'.format(save_dir)):
        result = np.loadtxt('{}/oligomers.txt'.format(save_dir))
        return result.T
    else:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        try:
            trace = np.loadtxt(trace_filename, delimiter=',', dtype='float')
        except ValueError:
            trace = np.loadtxt(trace_filename)

        trace_time = trace[-1,0]


        trace, bg_to_subtract = bg_function(trace[:,1], save_dir, bg_extra_args)

        fig, ax = plt.subplots()
        ax.plot(trace)
        ax.plot(bg_to_subtract, color='red')
        for cutoff in oligomer_cutoffs:
            ax.plot(bg_to_subtract+cutoff, color='xkcd:blue')
        ax.set_xlabel('Time /ms')
        ax.set_ylabel('Photons per ms')
        fig.tight_layout()
        fig.savefig('{}/trace_with_bg_cutoffs'.format(save_dir), bbox_inches='tight')
        plt.close()

        photons = []

        fig, axes = plt.subplots(nrows=len(oligomer_cutoffs), figsize=(3, 3 * len(oligomer_cutoffs)))
        ax = axes.ravel()

        for i, cutoff in enumerate(oligomer_cutoffs):
            ax[i].set_title('Cutoff: {}'.format(cutoff))

            to_analyse = np.copy(trace) - bg_to_subtract
            oligomers = to_analyse[to_analyse > cutoff]

            if np.size(oligomers) > 0:
                bins = np.arange(-0.5, np.amax(oligomers), 1)
                entries, bin_edges = np.histogram(oligomers, bins=bins, density=False)
                bin_middles = 0.5 * (bin_edges[1:] + bin_edges[:-1])
                ax[i].plot(bin_middles, entries)
                oligomer_photons = np.sum(entries)
                if as_frac_median:
                    oligomer_photons = oligomer_photons/np.sum(bg_to_subtract)
                else:
                    oligomer_photons = dilution_factor * oligomer_photons / trace_time

                photons.append(oligomer_photons)
            else:
                photons.append(0)
        fig.tight_layout()
        fig.savefig('{}/hists_oligomers'.format(save_dir), bbox_inches='tight')
        plt.close()

        result = np.vstack((oligomer_cutoffs, photons))
        np.savetxt('{}/oligomers.txt'.format(save_dir), result.T)

        return result

def analyse_step_scan(step_dir, bg_function, save_dir, oligomer_cutoffs, dilution_factor=1, load=True, bg_then_cut = True,
                      bg_extra_args = None, positions_range=[0,4000], as_frac_median=True):
    orig_save_dir = save_dir
    save_dir = '{}{}'.format(save_dir, step_dir[step_dir.rindex('/'):])
    overall_save_dir = save_dir

    print('overall savedir', overall_save_dir)

    if load:
        print('LOAD')
        if os.path.exists('{}/analysed_positions.txt'.format(overall_save_dir)) and\
            os.path.exists('{}/oligomer_photons.txt'.format(overall_save_dir)):
            positions = np.loadtxt('{}/analysed_positions.txt'.format(overall_save_dir))
            oligomer_photons = np.loadtxt('{}/oligomer_photons.txt'.format(overall_save_dir))
            return positions, oligomer_photons


    traces = glob.glob('{}/*csv'.format(step_dir))
    positions = []
    oligomer_photons = []
    for trace_filename in traces:
        try:
            pos = trace_filename[:trace_filename.rindex('um')]
            pos = float(pos[pos.rindex('_') + 1:])
            positions.append(pos)
        except:
            traces.remove(trace_filename)

    traces = [x for _, x in sorted(zip(positions, traces))]
    positions.sort()
    if len(traces) == 0:
        print(step_dir)
        pass

    used_positions = []
    for i,trace_filename in enumerate(traces):
        if positions[i] > positions_range[0] and positions[i] < positions_range[1]:
            used_positions.append(positions[i])
            oligomer_photons.append(get_oligomers(trace_filename, bg_function,
                                                  '{}{}'.format(save_dir, trace_filename[trace_filename.rindex('/'):-4]),
                                                  oligomer_cutoffs, dilution_factor = dilution_factor, bg_then_cut = bg_then_cut,
                                                              bg_extra_args = bg_extra_args, load_analysis=load,as_frac_median=as_frac_median)[1])
    oligomer_photons = np.array(oligomer_photons)

    colours = plt.cm.viridis(np.linspace(0,1,len(oligomer_cutoffs)))

    fig,ax = plt.subplots()
    for i in range(len(oligomer_cutoffs)):
        ax.plot(used_positions, oligomer_photons[:,i], label=oligomer_cutoffs[i], color = colours[i])

    ax.set_xlabel('Channel position /um')
    if as_frac_median:
        # output results as the fraction of photons in oligomers, relative to the background level
        ax.set_ylabel('Fraction of photons in oligomers')
    else:
        ax.set_ylabel('Oligomer photons per second')

    ax.legend(title = 'Oligomer cutoff')

    fig.savefig('{}/oligomers_mobilities'.format(overall_save_dir), bbox_inches='tight')
    plt.close()

    np.savetxt('{}/all_positions.txt'.format(overall_save_dir), positions)
    np.savetxt('{}/analysed_positions.txt'.format(overall_save_dir), used_positions)
    np.savetxt('{}/oligomer_photons.txt'.format(overall_save_dir), oligomer_photons)

    return used_positions, oligomer_photons

def create_config(filename, data_loc, output_filename):
    # Takes in the filename of a .txt file which contains a list of the subfolder names of the step_dirs
    file = np.loadtxt(filename, dtype='str', delimiter='\n')

    with open(output_filename, 'a', encoding='utf-8') as output_file:
        output_file.write('step_dir\taggregation time\tdilution factor')
        for row in file:
            output_file.write('\n{}/{}'.format(data_loc, row))
            agg_time = row[:row.rindex('h_')]
            agg_time = agg_time[agg_time.rindex('_') + 1:]
            if 'p' in agg_time:
                agg_time = agg_time.replace('p','.')
            output_file.write('\t{}'.format(agg_time))
            dilution = row[row.index('1in') + 3:]
            dilution = dilution[:dilution.index('_')]
            output_file.write('\t{}'.format(dilution))



def assemble_aggregation_trace(config_filename, bg_function, save_dir, oligomer_cutoffs, load=True, bg_then_cut = True,
                               bg_extra_args = None, dirs_to_analyse = None, positions_range=[0,4000], as_frac_median = True):
    config = np.loadtxt(config_filename, delimiter='\t', dtype='str', skiprows=1)
    if dirs_to_analyse is None:
        dirs_to_analyse = len(config)
    agg_times = []
    exp_dates = []
    FFE_nums = []
    sample_ids = []
    total_oligomer_photons = []
    for i,row in enumerate(config):
        if i < dirs_to_analyse:
            print(row[0])
            agg_times.append(float(row[1]))
            FFE_num = row[0][row[0].rindex('FFE') + 3:]
            FFE_num = int(FFE_num[:FFE_num.index('_')])
            FFE_nums.append(FFE_num)
            exp_date = row[0][row[0].rindex('/') + 1:]
            exp_date = int(exp_date[:exp_date.index('_')])
            exp_dates.append(exp_date)
            sample_ids.append('{}_FFE{}'.format(exp_date, FFE_num))
            positions, oligomer_photons = analyse_step_scan(row[0], bg_function, save_dir, oligomer_cutoffs,
                                                            dilution_factor=float(row[2]), load=load, bg_then_cut = bg_then_cut,
                                                            bg_extra_args = bg_extra_args, positions_range=positions_range,
                                                            as_frac_median = as_frac_median)
            # need to weight the positions by the distances between them
            position_diffs = [positions[i + 1] - positions[i] for i in range(len(positions) - 1)]
            position_diffs.append(position_diffs[-1])
            position_diffs = np.array(position_diffs)[:, np.newaxis]
            norm = len(positions) * np.multiply(position_diffs, oligomer_photons) / np.sum(position_diffs)
            total = np.mean(norm, axis=0)
            total_oligomer_photons.append(total)
    total_oligomer_photons = np.array(total_oligomer_photons)

    colours = plt.cm.viridis(np.linspace(0,1,len(oligomer_cutoffs)))
    fig,ax = plt.subplots()
    for i in range(len(oligomer_cutoffs)):
        ax.scatter(agg_times[:len(total_oligomer_photons)], total_oligomer_photons[:,i], label=oligomer_cutoffs[i], color=colours[i])

    ax.legend(title='Oligomer cutoffs')
    ax.set_xlabel('Aggregation time /h')
    ax.set_ylabel('Oligomer photons per second')
    fig.savefig('{}/agg_oligomers'.format(save_dir), bbox_inches='tight')

    ax.set_yscale('log')
    fig.savefig('{}/agg_oligomers_log'.format(save_dir), bbox_inches='tight')
    plt.close()

    agg_times_unique = list(set(agg_times))
    agg_times_unique.sort()

    np.savetxt('{}/agg_times.txt'.format(save_dir), agg_times_unique)

    agg_times_oligomers = {}

    for time in agg_times_unique:
        experiments = {}
        for i in range(np.amin([dirs_to_analyse, len(config)])):
            if agg_times[i] == time:
                if sample_ids[i] in experiments.keys():
                    experiments[sample_ids[i]] = np.vstack((experiments[sample_ids[i]], total_oligomer_photons[i]))
                else:
                    experiments[sample_ids[i]] = total_oligomer_photons[i]
        for exp in experiments.keys():
            if len(np.shape(experiments[exp])) > 1:
                experiments[exp] = np.median(experiments[exp], axis = 0)
        agg_times_oligomers[time] = experiments

    for time in agg_times_oligomers:
        print(time)
        exps = agg_times_oligomers[time]
        for exp in exps:
            print(exps[exp])

    agg_times_oligomer_medians = []
    for time in agg_times_oligomers.keys():
        data = agg_times_oligomers[time]
        data_for_median = np.array([data[i] for i in data.keys()])
        median = np.median(data_for_median, axis=0)
        agg_times_oligomer_medians.append(median)
    agg_times_oligomer_medians = np.array(agg_times_oligomer_medians)

    fig,ax = plt.subplots()
    for i in range(len(oligomer_cutoffs)):
        ax.plot(agg_times_unique, agg_times_oligomer_medians[:,i], label=oligomer_cutoffs[i], color=colours[i])
    ax.legend(title='Oligomer cutoffs')
    ax.set_xlabel('Aggregation time /h')
    ax.set_ylabel('Oligomer photons per second')
    fig.savefig('{}/agg_oligomers_medians'.format(save_dir), bbox_inches='tight')

    ax.set_yscale('log')
    fig.savefig('{}/agg_oligomers_medians_log'.format(save_dir), bbox_inches='tight')
    plt.close()

    print('assembly done')

def analyse_single_traces(config_filename, bg_function, oligomer_cutoffs,
                               bg_extra_args = None, combine = True, save_dir = 'combined'):
    config = np.loadtxt(config_filename, delimiter='\t', dtype='str', skiprows=1)
    all_oligos = []
    for row in config:
        a = get_oligomers(row[0], bg_function, row[1], oligomer_cutoffs, bg_extra_args=bg_extra_args)
        if combine:
            all_oligos.append(a)

    if combine:
        all_oligos = np.array(all_oligos)
        print(np.shape(all_oligos))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        colours = plt.cm.viridis(np.linspace(0, 1, len(oligomer_cutoffs)))
        fig,axes = plt.subplots(ncols=2, figsize=(8,4))
        ax = axes.ravel()
        for i,cutoff in enumerate(oligomer_cutoffs):
            ax[0].scatter(np.arange(1, len(config)+1), all_oligos[:, 1, i], color = colours[i])
            ax[1].scatter(np.arange(1, len(config)+1), all_oligos[:, 1, i], color=colours[i])
        ax[1].set_yscale('log')
        fig.savefig('{}/plot'.format(save_dir))
        plt.close()


def convert_to_molar(photons_per_sec, photons_per_monomer=22):
    # Convert to number of monomers/sec
    molar_conc = photons_per_sec/photons_per_monomer

    # Correct for the sampled volume
    # Units of metres
    spot_cross_section = 2e-7 * 1.5e-6 * math.pi
    channel_cross_section = 3000e-6 * 26e-6
    molar_conc = molar_conc*channel_cross_section/spot_cross_section

    # Take into account loss from desalting, assuming 40% loss
    molar_conc = molar_conc/0.6

    # Take into account the dilution on-chip due to flow rates
    # Units of uL/hr for the flow rates
    sample_flowrate = 10
    total_flowrate = 1150
    molar_conc = molar_conc*total_flowrate/sample_flowrate

    # This is now monomer mass going through the device per second, so need to look at the flow velocity
    # Use total_flowrate and channel_cross_section to calculate the flow velocity
    total_flowrate_uLs = total_flowrate/3600
    # 1 uL = 1e-9 m^3
    total_flowrate_m3s = total_flowrate_uLs * 1e-9
    # total_flowrate_ms = total_flowrate_m3s/channel_cross_section
    molar_conc = molar_conc/total_flowrate_m3s
    # So molar_conc is now units of number of monomers per m3, so divide by 1000 to get number of monomers per litre
    molar_conc = molar_conc/1000
    # Now it's in units of number of monomers per litre, so now convert to molar
    molar_conc = molar_conc/(6.022e23)
    return molar_conc









