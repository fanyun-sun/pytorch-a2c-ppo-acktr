import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from glob import glob
import pandas as pd
from scipy.interpolate import spline

def plot_reward(dir_names, legends):

    for idx, dir_name in enumerate(dir_names):
        files = glob('{}/*.monitor.csv'.format(dir_name))
        res = pd.read_csv(files[0], skiprows=1)['r'].values[:3000]
        for f in files[1:]:
            res += pd.read_csv(f, skiprows=1)['r'].values[:3000]

        res /= len(files)
        plt.plot(pd.rolling_mean(res, window=100), label=legends[idx])
    plt.legend()
    plt.savefig('reward.png')
    plt.close()

def plot_saturation(fnames, legends):
    for idx, fname in enumerate(fnames):
        df = pd.read_csv(fname, header=None)
        plt.plot(pd.rolling_mean(df.iloc[:25000, 0], window=100), label='{}-relu1'.format(legends[idx]))
        plt.plot(pd.rolling_mean(df.iloc[:25000, 1], window=100), label='{}-relu2'.format(legends[idx]))
    plt.legend()
    plt.savefig('saturation.png')
    plt.close()

if __name__ == '__main__':

    plot_reward(['tmp1000000000000', 'tmp20000', 'tmp5000'], ['original', 'scale-20000', 'scale-5000'])
    plot_saturation(['tmp1000000000000.sat', 'tmp20000.sat', 'tmp5000.sat'], ['original', 'scale-20000', 'scale-5000'])
