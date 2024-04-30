import readPTU as rp
import os
import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import argparse
import glob


timetrace_resolution = 1e-3   # in seconds
g2_resolution = 1e-3  # picoseconds * 1e-12 to change to seconds
g2_window = 30   # picoseconds * 1e-12 to change to seconds
threshold = 10 # in unit of counts per unit timetrace resolution
constraint_above = True # post-selection counts > threshold for the given timetrace resolution

parser = argparse.ArgumentParser()
requiredNamed = parser.add_argument_group('required named arguments')
requiredNamed.add_argument('-f', '--files_to_convert', type=str, help='filename of file containing directories to convert', required=True)
args = parser.parse_args()

ptu_dirs, csv_dirs = np.genfromtxt(args.files_to_convert,skip_header=1,dtype='str',delimiter='\t').T
for dir in ptu_dirs:
    if not os.path.exists(dir):
        print('Ptu directory {} does not exist. Terminating.'.format(dir))
        exit()

for i,ptu_dir in enumerate(ptu_dirs):
    if not os.path.exists(csv_dirs[i]):
        os.makedirs(csv_dirs[i])
    if not os.path.exists(csv_dirs[i]+'_plots'):
        os.makedirs(csv_dirs[i]+'_plots')
    ptu_files = glob.glob(ptu_dir + '/*um_steps.ptu')
    for file in ptu_files:
        position = file[:file.rindex('_') - 2]
        position = position[position.rindex('_') + 1:]
        with rp.PTUfile(file) as ptu_file:
            ptu_meas = rp.PTUmeasurement(ptu_file)
            timetrace_x, timetrace_y, timetrace_recnum = ptu_meas.timetrace(resolution=timetrace_resolution, record_range=[0,None])
            timetrace = np.vstack((timetrace_x,timetrace_y))
            np.savetxt(csv_dirs[i]+'/timetrace{}res_{}.csv'.format(str(timetrace_resolution),position),timetrace.T, delimiter=',')

            fig, ax = plt.subplots()
            ax.plot(timetrace_x, timetrace_y)
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Counts/{} s'.format(timetrace_resolution))
            ax.set_title('Timetrace')
            fig.savefig(csv_dirs[i]+'_plots'+'/timetrace{}res_{}.svg'.format(str(timetrace_resolution),position))
            plt.close()
