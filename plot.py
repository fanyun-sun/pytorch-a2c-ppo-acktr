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

from pylab import rcParams
print(rcParams['figure.figsize'])
rcParams['figure.figsize'] = 8, 4
markers = ['o', 'v', 's', '^', 'D']


def plot_reward(dir_names, legends, title=None, num=None, shade=False):

    sns.color_palette('hls', 10)
    x_label = 'million frames'
    y_label = 'cumulative reward'
    legend_title = hue = 'reward scaling'

    dfss = []
    for idx, dir_name in enumerate(dir_names):
        files = glob('{}/*.monitor.csv'.format(dir_name))

        minlen = num
        # if num is not None:
            # minlen = num
        # else:
            # minlen = min([pd.read_csv(f, skiprows=1).shape[0] for f in files])

        print(dir_name, num)
        if shade:
            dfs = []
            for f in files:
                res = pd.read_csv(f, skiprows=1)['r'].values[:minlen]
                res = pd.rolling_mean(res, window=500)

                df = pd.DataFrame()
                df[y_label] = res
                df[x_label] = np.arange(res.shape[0])
                df[hue] = legends[idx]
                dfs.append(df)

            dfs = pd.concat(dfs)
            dfss.append(dfs)
        else:
            for f in files[:1]:
                tmpdf = pd.read_csv(f, skiprows=1)  

                # df[y_label] = pd.rolling_mean(tmpdf['r'].values[:minlen], window=500)
                # df[x_label] = tmpdf['l'].cumsum().values[:minlen]

                y = pd.rolling_mean(tmpdf['r'], window=500).values
                x = tmpdf['l'] = tmpdf['l'].cumsum().values
                

                if num is not None:
                    x = tmpdf[tmpdf.l < num]['l']
                    y = y[:x.shape[0]]
                
                x /= 1000000
                print(x.shape, y.shape)
                plt.plot(x, y, label=legends[idx])


    # df = pd.concat(dfss)
    # df = df.append({x_label:0, y_label:0, hue:'dummy'}, ignore_index=True)
    # g = sns.lineplot(x=x_label, y=y_label, hue=hue, data=df, style=hue)
    # g = sns.lineplot(x=x_label, y=y_label, hue=hue, style=hue, markers=['o', 'v', 's', 'p'], dashes=False, data=df, markevery=100)
    # g = sns.lineplot(x=x_label, y=y_label, hue=hue, style=hue, markers=True, dashes=False, data=df, markevery=100)

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.gcf().subplots_adjust(bottom=0.15)

    ax = plt.gca()
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height]) # resize position

    if title is not None:
        plt.title(title)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., title=legend_title)

    # plt.tight_layout()

    plt.savefig('reward.png')

    plt.close()

def incremental(x):
    print(x)
    beta = .9
    ret = np.zeros(len(x))
    ret[0] = x[0]
    for i in range(1, ret.shape[0]):
            ret[i] = (beta * ret[i-1] + (1-beta) * x[i]) / (1-beta ** i)
    print(ret)
    return ret


def plot_saturation(fnames, legends, title=None, num=None):
    fnames = ['{}/{}.sat'.format(x, os.path.basename(x)) for x in fnames]
    x_label = 'million frames'
    y_label = 'PDRR'
    legend_title= 'reward scaling'
    def go(layeridx):
        for idx, fname in enumerate(fnames):
            df = pd.read_csv(fname, header=None)
            if layeridx == 2:
                y = pd.rolling_mean(df.iloc[:, 1] ,window=500)[:num]
                plt.plot(np.arange(y.shape[0])*10/1e6 * 5, y, label='{}-relu{}'.format(legends[idx], layeridx))
            else:
                y = pd.rolling_mean(df.iloc[:, 0] ,window=500)[:num]
                plt.plot(np.arange(y.shape[0])*10/1e6 * 5, y, label='{}-relu{}'.format(legends[idx], layeridx))

        if title is not None:
            plt.title(title)

        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.gcf().subplots_adjust(bottom=0.15)

        ax = plt.gca()
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height]) # resize position

        if title is not None:
            plt.title(title)
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., title=legend_title)

        # plt.tight_layout()
        plt.savefig(f'relu{layeridx}')
        plt.close()
    go(2)
    go(1)

def plot_dirs(dirs, legends=None):
    if legends is not None:
        plot_reward(dirs, legends)
        plot_saturation(dirs, legends)
    else:
        plot_reward(dirs, dirs)
        plot_saturation(dirs, dirs)

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
    # dirs = glob('Hopper-v2-*')
    # for f in glob('Hopper-v2*lr*'):
        # dirs.remove(f)

    # dirs = glob('*reward*')
    # dirs += ['../Hopper-v2/network_64', '../Hopper-v2/Hopper-v2-network_64-scale_thresh_.2']

    # plot_dirs(dirs)

    # plot_dirs(glob('Hopper-v2-network_64*') + ['../Hopper-v2/network_64'], None)

    # plot_dirs(glob('HalfCheetah-v2-network_64*'))

    # dirs = ['./Hopper-v2-network_64-network_ratio_.5-reward_scaling_1', 'Hopper-v2-network_64-network_ratio_1.-adam', './Hopper-v2-network_64-network_ratio_.1-rmsprop_rescaling', '../Hopper-v2/Hoper-network_6']
    # legends = ['rmsprop', 'adam', 'rmsprop scaling']
    # plot_reward(dirs,legends, title='Hopper-v2', num=None)
    # plot_saturation(dirs, legends)
    #plot_dirs(glob('../Swimmer-v2/*'))
    # dirs = ['Hopper-v2-network_64-network_ratio_.5-reward_scaling_1']+glob('Hopper-v2-network_64-network_ratio_.5-lr*')

    # legends = ['lr 4e-7', 'lr '+ dirs[1][-4:], 'lr ' + dirs[2][-4:]]
    # legends = [x[-4:] for x in dirs]

    # plot_reward(dirs, legends)
    # plot_saturation(dirs, legends)
    """
    # dirs =glob('Hopper-v2-network_8*')
    # dirs.remove('Hopper-v2-network_8-network_ratio_.5-reward_scaling_.25')
    # dirs.remove('Hopper-v2-network_8-network_ratio_.5-reward_scaling_.5')
    # dirs.remove('Hopper-v2-network_8-network_ratio_.5-reward_scaling_.25')
    # plot_dirs(dirs)
    """

    # pref = './Hopper-v2-network_64-network_ratio_.5-reward_scaling_'
    # scales =  ['.5', '.75', '1', '5', '10', '30', '100']
    # dirs = [f'{pref}{x}' for x in scales]
    # legends = scales
    # plot_reward(dirs, legends, num=1500000)
    # plot_saturation(dirs, legends, num=30000)

    pref = './HalfCheetah-v2-network_64-network_ratio_1.-reward_scaling-'
    scales =  ['.75', '5', '30']
    dirs = [f'{pref}{x}' for x in scales]
    legends = scales
    plot_reward(dirs, legends, num=1500000)
    plot_saturation(dirs, legends, num=30000)

    # pref = 'Hopper-v2-network_64-network_ratio_.5-lr_'
    # lrs = ['1e-5', '1e-4', '7e-4', '1e-3', '7e-3' ]
    # dirs = ['{}{}'.format(pref, lr) for lr in lrs]
    # dirs[2] = 'Hopper-v2-network_64-network_ratio_.5-reward_scaling_1'
    # plot_reward(dirs, lrs, num=1200000) 
    # plot_saturation(dirs, lrs, num=25000)

    # dirs = ['Hopper-v2-network_64-network_ratio_.1-scaling_leaky', 'Hopper-v2-network_64-network_ratio_.1-rmsprop_rescaling', 'Hopper-v2-network_64-weight_.1']
    # legends = ['leaky + out method', 'our method', 'original']
    # plot_reward(dirs, legends)

