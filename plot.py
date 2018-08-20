import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from glob import glob
import pandas as pd
from scipy.interpolate import spline
import os
import numpy as np

import seaborn as sns; sns.set()
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def plot_reward(dir_names, legends, title='Hopper-v2', num=2500, shade=False):

    sns.color_palette('hls', 10)
    x_label = 'number of episodes'
    y_label = 'cumulative reward'
    hue = 'reward scaling'

    dfss = []
    for idx, dir_name in enumerate(dir_names):
        files = glob('{}/*.monitor.csv'.format(dir_name))
        if num is not None:
            minlen = num
        else:
            minlen = min([pd.read_csv(f, skiprows=1).shape[0] for f in files])

        print(dir_name, minlen)
        if shade:
            dfs = []
            for f in files:
                res = pd.read_csv(f, skiprows=1)['r'].values[:minlen]
                res = pd.rolling_mean(res, window=20)

                df = pd.DataFrame()
                df[y_label] = res
                df[x_label] = np.arange(res.shape[0])
                df[hue] = legends[idx]
                dfs.append(df)

            dfs = pd.concat(dfs)
            dfss.append(dfs)
        else:
            res = pd.read_csv(files[0], skiprows=1)['r'].values[:minlen]
            for f in files[1:]:
                res += pd.read_csv(f, skiprows=1)['r'].values[:minlen]
            res /= len(files)
            res = pd.rolling_mean(res, window=20)
            df = pd.DataFrame()
            df[y_label] = res
            df[x_label] = np.arange(res.shape[0])
            df[hue] = legends[idx]
            dfss.append(df)

    
    df = pd.concat(dfss)
    df[hue] = df[hue].astype(str)

    sns.lineplot(x=x_label, y=y_label, hue=hue, data=df)

    if title is not None:
        plt.title(title)
    plt.savefig('reward.png')
    plt.close()

def incremental(x):
    beta = .5
    ret = np.zeros(x.shape)
    ret[0] = x[0]
    for i in range(1, ret.shape[0]):
        ret[i] = beta * ret[i-1] + (1-beta) * x[i]
    return ret


def plot_saturation(fnames, legends, title=None, num=24000):
    fnames = ['{}/{}.sat'.format(x, os.path.basename(x)) for x in fnames]
    xlabel = 'number of updates'
    ylabel = 'PDRR'
    for idx, fname in enumerate(fnames):
        df = pd.read_csv(fname, header=None)
        # plt.plot(pd.rolling_mean(df.iloc[:, 0][:num], window=100), label='{}-relu1'.format(legends[idx]))
        # plt.plot(incremental(df.iloc[:, 0][:num]), label='{}-relu1'.format(legends[idx]))

        plt.plot(pd.rolling_mean(df.iloc[:, 1][:num], window=100), label='{}-relu2'.format(legends[idx]))
    if title is not None:
        plt.title(title)
    plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig('saturation.png')
    plt.close()

def plot_dirs(dirs):
    print(dirs)
    plot_reward(dirs, dirs)
    plot_saturation(['{}/{}.sat'.format(x, os.path.basename(x)) for x in dirs], dirs)

if __name__ == '__main__':


    # Hopper different scaling interval experiment
    # plot_reward(['../Hopper-v2/Hopper-v2-scale-5000', '../Hopper-v2/network_64', 'grad_clipping'], ['scale_5000', 'original', 'grad_clipping'], 2300)
    # plot_saturation(['../Hopper-v2/Hopper-v2-scale-5000/Hopper-v2-scale-5000.sat', '../Hopper-v2/network_64/network_64.sat', 'grad_clipping.sat'], ['scale-5000', 'original', 'grad_clipping'], num=25000)

    ## Hopper network size saturation experiment
    # plot_reward(['./Hopper-v2-network_8', '../Hopper-v2/network_16', '../Hopper-v2/network_64', '../Hopper-v2/network_256', 'Hopper-v2-network_8-scale_thresh_.4'], ['network_8', 'network_16', 'network_64', 'network_256', 'network_8-scale_thresh_.2'])
    # plot_saturation(['../Hopper-v2/network_16/network_16.sat', '../Hopper-v2/network_64/network_64.sat', '../Hopper-v2/network_256/network_256.sat'], ['network_16', 'network_64', 'network_256'])

    # Hopper small network v.s. out method
    # plot_reward(['./Hopper-v2-network_8', 'Hopper-v2-network_8-scale_thresh_.4'], ['network_8', 'network_8-scale_thresh_.2'])
    # plot_saturation(['./Hopper-v2-network_8.sat', 'Hopper-v2-network_8-scale_thresh_.4.sat'], ['network_8', 'network_8-scale_thresh_.2'])

    # Hopper, with v.s. without gradient clipping
    # plot_reward(['../Hopper-v2/Hopper-v2-scale-5000', 'grad_clipping', '../Hopper-v2/grad_clipping', '../Hopper-v2/network_64'], ['scale', 'scale+clipping', 'grad clipping', 'original'], 2300)

    # HalfCheetah, with v.s. without gradient clipping
    # plot_directory('../HalfCheetah-v2')
    # plot_reward(['grad_clipping-scale_20000-HalfCheetah-v2','../HalfCheetah-v2/grad_clipping', '../HalfCheetah-v2/original', '../HalfCheetah-v2/scale_20000'], ['clipping+scaling', 'grad_clipping', 'original', 'scale_20000'], 1600)
    # plot_saturation(['grad_clipping1.sat', '../HalfCheetah-v2/original/original.sat', '../HalfCheetah-v2/scale_20000/scale_20000.sat'], ['grad_clipping', 'original', 'scale_20000'])

    # Hopper, different reward scaling(no other tricks used)
    """
    name = '../Hopper-v2/Hopper-v2-rewrard_scale-'
    scales = ['.5', '1', '5', '10']
    plot_reward(['{}{}'.format(name, s) for s in scales], ['reward_scale-{}'.format(x) for x in scales])
    tmp = 'Hopper-v2-rewrard_scale-'
    plot_saturation(['{}{}/{}{}.sat'.format(name, s, tmp, s) for s in scales], ['reward_scale-{}'.format(x) for x in scales])
    """
    # plot_directory('../Hopper-v2')
    # dirs = ['Hopper-v2-network_64-scale_thresh_.2', '../Hopper-v2/network_64']
    # plot_reward(dirs, dirs, 'Hopper-v2')
    # plot_reward(dirs, [os.path.basename(d) for d in dirs], num=6000)
    # sat = ['{}/{}.sat'.format(x, os.path.basename(x)) for x in dirs]
    # plot_saturation(sat, [os.path.basename(d) for d in dirs], num=20000)

    # x = ['HalfCheetah-v2-network_8-scale_thresh_.2', 'HalfCheetah-v2-network_8', '../HalfCheetah-v2/HalfCheetah-v2-network_8-scale_thresh_.2-single', 'HalfCheetah-v2-network_64-scale_thresh_.4']
    # dirs = ['../HalfCheetah-v2/HalfCheetah-v2-network_64-scale_thresh_.2', '../HalfCheetah-v2/original', 'HalfCheetah-v2-network_64-scale_thresh_.4', '../HalfCheetah-v2/scale_20000']
    """
    dirs = ['HalfCheetah-v2-network_8-scale_thresh_.2-20000', 'HalfCheetah-v2-network_8-scale_thresh_.4-20000', '../HalfCheetah-v2/HalfCheetah-v2-network_8']

    plot_reward(dirs, dirs, title='HalfCheetah-v2')
    sat = ['{}/{}.sat'.format(x, os.path.basename(x)) for x in dirs]
    sat[-2] = 'HalfCheetah-v2-network_64-scale_thresh_.4.sat'
    plot_saturation(sat, [os.path.basename(x) for x in dirs], title='HalfCheetah-v2')
    """

    # x = ['../Ant-v2/Ant-v2-network_8', '../Ant-v2/Ant-v2-network_8-scale_thresh_.2', 'Ant-v2-network_64-scale_thresh_.2']
    # plot_reward(x, x)

    # plot_saturation([xx + '.sat' for xx in x], x)

    # x = ['../Humanoid-v2/Humanoid-v2-network_8', '../Humanoid-v2/Humanoid-v2-network_8-scale_thresh_.2']
    # plot_reward(x, x)
    # plot_saturation(['{}/{}.sat'.format(xx, os.path.basename(xx)) for xx in x], x)

    # PPO
    # plot_directory('../../ppo/Hopper-v2')

    # Walker
    # plot_directory('../Walker-2d-v2')
    # plot_dirs(glob('HalfCheetah-v2-network_64*') + ['../HalfCheetah-v2/grad_clipping'])
    # plot_dirs(glob('../Hopper-v2/Hopper-v2-rewrard_scale*'))
    pref = '../Hopper-v2/Hopper-v2-rewrard_scale-'
    plot_reward([f'{pref}{x}' for x in ['.5', '1', '5', '10']]+ ['Hopper-v2-network_64-rewrard_scale-20'], ['ratio .5', 'ratio 1', 'ratio 5', 'ratio 10', 'ratio 20'])
    plot_saturation([f'{pref}{x}' for x in ['.5', '1', '5', '10']] + ['Hopper-v2-network_64-rewrard_scale-20'], ['ratio .5', 'ratio 1', 'ratio 5', 'ratio 10', 'ratio 20'])

