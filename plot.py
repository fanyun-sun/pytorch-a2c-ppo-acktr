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
from scipy.interpolate import interp1d
import pandas as pd

from pylab import rcParams
print(rcParams['figure.figsize'])
# rcParams['figure.figsize'] = 8, 4
# markers = ['o', 'v', 's', '^', 'D']
legend_title = 'learning rate'


def plot_reward(dir_names, legends, fname='reward.png', title=None, num=None, shade=True):

    sns.color_palette('hls', 10)
    x_label = 'million frames'
    y_label = 'cumulative reward'
    hue = legend_title

    dfss = []
    for idx, dir_name in enumerate(dir_names):
        files = glob('{}/*.monitor.csv'.format(dir_name))

        # if num is not None:
            # minlen = num
        # else:
            # minlen = min([pd.read_csv(f, skiprows=1).shape[0] for f in files])

        if shade:
            for f in files[:1]:
                tmpdf =pd.read_csv(f, skiprows=1) 

                x = tmpdf['l'].cumsum().values * 16
                if 'Swimmer-v2' in title:
                    y = pd.rolling_mean(tmpdf['r'].values, window=10)
                else:
                    y = pd.rolling_mean(tmpdf['r'].values, window=100)

                if f == files[0]:
                    print(dir_name, x[-1])


                tmpf = interp1d(x, y)
                df = pd.DataFrame()

                interval = 50000
                lowx = (x[0] + interval - 1)//interval * interval
                df[x_label] = np.arange(lowx , 9980000, interval) 
                df[y_label] = tmpf(df[x_label].values)
                df[x_label] /= 1e6

                df[hue] = legends[idx]
                dfss.append(df)
            
        else:
            for f in files[:1]:
                tmpdf = pd.read_csv(f, skiprows=1)  

                # df[y_label] = pd.rolling_mean(tmpdf['r'].values[:minlen], window=500)
                # df[x_label] = tmpdf['l'].cumsum().values[:minlen]

                y = pd.rolling_mean(tmpdf['r'], window=100).values
                x = tmpdf['l'] = tmpdf['l'].cumsum().values * 16
                
                if f == files[0]:
                    print(dir_name, x[-1])

                tmpf = interp1d(x, y)

                interval = 50000
                lowx = (x[0] + interval - 1)//interval * interval

                x = np.arange(lowx , min(x[-1], 9980000), interval) 
                x = x.astype(np.float32)
                y = tmpf(x)
                x /= 1e6

                plt.plot(x, y, label=legends[idx])


    if shade:
        """
        if title == 'Walker2d-v2':
            print(plt.ylim())
            ymin, ymax = plt.ylim()
            ax = plt.gca()
            ax.set_ylim([-1000, 3000])
            print(plt.ylim())
        if title == 'HalfCheetah-v2':
            print(plt.ylim())
            ymin, ymax = plt.ylim()
            ax = plt.gca()
            ax.set_ylim([-1000, 7000])
            print(plt.ylim())
        """
        df = pd.concat(dfss)
        print('plotting...')
        g = sns.lineplot(x=x_label, y=y_label, hue=hue, err_kws={'alpha': 0.1}, data=df)
        ax = plt.gca()
        # box = ax.get_position()
        # ax.set_position([box.x0, box.y0, box.width * 0.8, box.height]) # resize position
        # plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)


        handles, labels = ax.get_legend_handles_labels()
        legend =ax.legend(handles=handles[1:], labels=labels[1:], loc=2)
        plt.setp(legend.get_title(),fontsize='xx-small')



    else:
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.legend()


    plt.gcf().subplots_adjust(bottom=0.15)
    if title is not None:
        plt.title(title)

    # plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., title=legend_title)
    # plt.tight_layout()
    plt.savefig(fname)


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


def plot_saturation(fnames, legends, fname='relu', shade=True, title=None, num=None):

    sns.color_palette('hls', 10)
    hue = legend_title
    x_label = 'million frames'
    y_label = 'PDRR'
    fnames = [glob('{}/*.sat'.format(x))[0] for x in fnames]

    def go(layeridx):

        dfss = []
        for idx, f in enumerate(fnames):

            tmpdf =pd.read_csv(f, header=None) 

            if layeridx == 2:
                y = pd.rolling_mean(tmpdf.iloc[:, 1] ,window=100)
                x = np.arange(y.shape[0]) * 10 * 5 * 16
            else:
                y = pd.rolling_mean(tmpdf.iloc[:, 0] ,window=100)
                x = np.arange(y.shape[0]) * 10 * 5 * 16
                


            tmpf = interp1d(x, y)
            df = pd.DataFrame()

            interval = 50000
            lowx = (x[0] + interval-1)//interval* interval
            print(x[-1])
            df[x_label] = np.arange(lowx , 9999000, interval) 
            df[y_label] = tmpf(df[x_label].values)
            df[x_label] /= 1e6

            df[hue] = legends[idx]
            dfss.append(df)
            
        df = pd.concat(dfss)
    # df = df.append({x_label:0, y_label:0, hue:'dummy'}, ignore_index=True)
        print('plotting ...')
        g = sns.lineplot(x=x_label, y=y_label, hue=hue, data=df, err_kws={'alpha': 0.1})

        plt.gcf().subplots_adjust(bottom=0.15)
        # ax = plt.gca()
        # box = ax.get_position()
        # ax.set_position([box.x0, box.y0, box.width * 0.8, box.height]) # resize position

        plt.title(title)

        ax = plt.gca()
        handles, labels = ax.get_legend_handles_labels()
        legend =ax.legend(handles=handles[1:], labels=labels[1:], loc=2)

        # plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., title=legend_title)
        # plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        # plt.tight_layout()
        plt.savefig('{}-relu{}.png'.format(fname, layeridx))

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

    # leaky, elu experiment
    """
    title = 'Hopper-v2 (A2C)'
    legend_title = 'activation'
    pref = '../Hopper-v2/Hopper-v2-network_64-network_ratio_.5-reward_scaling-'
    for scale in ['.5', '1.', '10' ,'30', '50', '70']:
        if scale == '1.':
            dirs = glob(pref + scale + '*') +  glob(pref + scale[0] + '-' + '*')
        else:
            dirs = glob(pref + scale + '*')

        dirs.sort()
        legends = []
        for d in dirs:
            if 'leaky' in d:
                legends.append('leaky-relu')
            elif 'elu' in d:
                legends.append('elu')
            else:
                legends.append('relu')

        print(legends)
        assert len(legends) == 9
        plot_reward(dirs, legends, title=title, fname='Hopper-v2-scale-{}.png'.format(scale))

    dirs = []
    for scale in ['.5', '50']:
        dirs += glob(pref + scale + '*')

    dirs.sort()
    legends = []
    for d in dirs:
        if 'leaky' in d:
            leg = 'leaky-relu'
        elif 'elu' in d:
            leg = 'elu'
        else:
            leg = 'relu'

        if '50' in d:
            leg += ' (50)'
        else:
            leg += ' (.5)'
        legends.append(leg)


    print(legends)
    assert len(legends) == 18

    # title = '{}(reward scaling {})'.format('Hopper-v2', scale)
    plot_reward(dirs, legends, title=title, fname='activation-.5-50.png')
    """

    # learning rate experiment
    """
    legend_title = 'lr'
    dirs = glob('../Hopper-v2/lr_exp/*lr_*')
    dirs.sort()
    assert len(dirs) == 5 * 3
    legends = [x[-4:] for x in dirs]
    print(legends)
    plot_reward(dirs, [x + '_' for x in legends], fname='lr.png', title='Hopper-v2')
    plot_saturation(dirs, [x + '_' for x in legends], fname='lr', title='Hopper-v2')
    """
    
    # reward scalling experiment
    """
    for env in [ 'Hopper-v2', 'Walker2d-v2', 'Ant-v2', 'Walker2d-v2', 'HalfCHeetah-v2']:
        # if env == 'Swimmer-v2' or env == 'Walker2d-v2' or env == 'Ant-v2':
            # continue
    # env = 'Ant-v2'

        legend_title = 'reward scale'
        dirs = glob('../{}/*network_ratio_.5*reward_scaling*'.format(env))
        dirs = [d for d in dirs if 'elu' not in d and 'leaky' not in d]
        print(dirs)
        dirs.sort()
        print(len(dirs))
        # assert len(dirs) == 6 * 3
        legends = []
        for d in dirs:
            if 'seed_1' in d or 'seed_2' in d or 'seed_3' in d:
                legends.append(d[-9:-7])
            else:
                legends.append(d[-2:])
        legends = [x[1] + '.' if x=='-1' else x for x in legends]
        print(legends)
        assert len(set(legends)) == 6
        plot_reward(dirs, [x+'_' for x in legends], fname='{}-reward_scaling.png'.format(env), title=env)
        plot_saturation(dirs, [x+'_' for x in legends], fname=env, title=env)
    """

    # our method
    dirs = glob('Swimmer*') + ['../Swimmer-v2/Swimmer-v2-network_64-network_ratio_1.-reward_scaling-1.']
    legends = [d[-10:] for d in dirs]
    legends[-1] = 'original'
    plot_reward(dirs, legends, fname='Swimmer-v2', shade=False)

    dirs = glob('Walker2d*') + ['../Walker2d-v2/Walker2d-v2-network_64-network_ratio_1.-reward_scaling-1.']
    legends = [d[-10:] for d in dirs]
    legends[-1] = 'original'
    plot_reward(dirs, legends, num=1000, shade=False, fname='Walker2d-v2')

    dirs = glob('Hopper-v2*') + ['../Hopper-v2/Hopper-v2-network_64-network_ratio_1.-reward_scaling-1-seed_1']
    legends = [d[-10:] for d in dirs]
    legends[-1] = 'original'
    plot_reward(dirs, legends, num=1000, shade=False, fname='Hopper-v2')

    dirs = glob('HalfCheetah-v2*') + ['../HalfCheetah-v2/HalfCheetah-v2-network_64-network_ratio_1.-reward_scaling-1.']
    legends = [d[-10:] for d in dirs]
    legends[-1] = 'original'
    plot_reward(dirs, legends, num=1000, shade=False, fname='HalfCheetah=v2')

    dirs = glob('Ant-v2*') + ['../Ant-v2/Ant-v2-network_64-network_ratio_1.-reward_scaling-1']
    legends = [d[-10:] for d in dirs]
    legends[-1] = 'original'
    plot_reward(dirs, legends, num=1000, shade=False, fname='Ant-v2')
