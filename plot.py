import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from glob import glob
import pandas as pd
from scipy.interpolate import spline
import os

def plot_reward(dir_names, legends, num=None):

    for idx, dir_name in enumerate(dir_names):
        print(dir_name)
        files = glob('{}/*.monitor.csv'.format(dir_name))
        if num is not None:
            minlen = num
        else:
            minlen = min([pd.read_csv(f, skiprows=1).shape[0] for f in files])
        res = pd.read_csv(files[0], skiprows=1)['r'].values[:minlen]
        for f in files[1:]:
            res += pd.read_csv(f, skiprows=1)['r'].values[:minlen]

        res /= len(files)
        plt.plot(pd.rolling_mean(res, window=10), label=legends[idx])
    plt.legend()
    plt.savefig('reward.png')
    plt.close()

def plot_saturation(fnames, legends, num=1000000000000):
    for idx, fname in enumerate(fnames):
        df = pd.read_csv(fname, header=None)
        # plt.plot(pd.rolling_mean(df.iloc[:, 0][:num], window=100), label='{}-relu1'.format(legends[idx]))

        plt.plot(pd.rolling_mean(df.iloc[:, 1][:num], window=100), label='{}-relu2'.format(legends[idx]))
    plt.legend()
    plt.savefig('saturation.png')
    plt.close()

def plot_directory(dirname):
    dirs = glob('{}/*'.format(dirname))
    print(dirs)
    plot_reward(dirs, dirs)
    plot_saturation(['{}/{}.sat'.format(x, os.path.basename(x)) for x in dirs], dirs)

if __name__ == '__main__':


    # Hopper different scaling interval experiment
    # plot_reward(['../Hopper-v2/Hopper-v2-scale-5000', '../Hopper-v2/network_64', 'grad_clipping'], ['scale_5000', 'original', 'grad_clipping'], 2300)
    # plot_saturation(['../Hopper-v2/Hopper-v2-scale-5000/Hopper-v2-scale-5000.sat', '../Hopper-v2/network_64/network_64.sat', 'grad_clipping.sat'], ['scale-5000', 'original', 'grad_clipping'], num=25000)

    ## Hopper network size saturation experiment
    # plot_reward(['../Hopper-v2/network_16', '../Hopper-v2/network_64', '../Hopper-v2/network_256'], ['network_16', 'network_64', 'network_256'], num=2500)
    # plot_saturation(['../Hopper-v2/network_16/network_16.sat', '../Hopper-v2/network_64/network_64.sat', '../Hopper-v2/network_256/network_256.sat'], ['network_16', 'network_64', 'network_256'])

    # Hopper, with v.s. without gradient clipping
    # plot_reward(['../Hopper-v2/Hopper-v2-scale-5000', 'grad_clipping', '../Hopper-v2/grad_clipping', '../Hopper-v2/network_64'], ['scale', 'scale+clipping', 'grad clipping', 'original'], 2300)

    # HalfCheetah, with v.s. without gradient clipping
    # plot_directory('../HalfCheetah-v2')
    # plot_reward(['grad_clipping-scale_20000-HalfCheetah-v2','../HalfCheetah-v2/grad_clipping', '../HalfCheetah-v2/original', '../HalfCheetah-v2/scale_20000'], ['clipping+scaling', 'grad_clipping', 'original', 'scale_20000'], 1600)
    # plot_saturation(['grad_clipping1.sat', '../HalfCheetah-v2/original/original.sat', '../HalfCheetah-v2/scale_20000/scale_20000.sat'], ['grad_clipping', 'original', 'scale_20000'])

    # Hopper, different reward scaling(no other tricks used)
    name = '../Hopper-v2/Hopper-v2-rewrard_scale-'
    scales = ['.5', '1', '5', '10']
    plot_reward(['{}{}'.format(name, s) for s in scales], ['reward_scale-{}'.format(x) for x in scales]) 
    tmp = 'Hopper-v2-rewrard_scale-'
    plot_saturation(['{}{}/{}{}.sat'.format(name, s, tmp, s) for s in scales], ['reward_scale-{}'.format(x) for x in scales]) 
