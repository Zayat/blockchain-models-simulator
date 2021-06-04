import colorsys
from sys import platform as sys_pf
if sys_pf == 'darwin':
    import matplotlib
    matplotlib.use("TkAgg")
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import pandas as pd
from glob import glob
import os
from graphviz import Graph
import humanfriendly

from utils import ApplicationPaths, LoggerFactory
'''
colors = {'red': '#e41a1c', 'blue': '#377eb8',
          'green': '#4daf4a', 'grey': '#404040'}

font = 'Clear Sans'
plt.rcParams["figure.figsize"] = [8.5, 4.5]
plt.rcParams['font.family'] = font
plt.rcParams['font.sans-serif'] = font
plt.style.use('fivethirtyeight')
plt.rcParams['axes.linewidth'] = 1
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False
plt.rcParams['ytick.color'] = 'black'
plt.rcParams['xtick.color'] = 'black'
plt.rcParams['axes.edgecolor'] = colors['grey']
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['savefig.facecolor'] = 'white'
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['xtick.major.size'] = 10
plt.rcParams['xtick.minor.size'] = 6
plt.rcParams['ytick.major.size'] = 10
plt.rcParams['ytick.minor.size'] = 6
plt.rcParams['font.size'] = 20
pd.set_option('precision', 8)
'''
logger_simulator = LoggerFactory.get_logger("logger_simulator")


def shorten_duration_string(str):
    s = str
    s = s.replace(" day", "d")
    s = s.replace(" hour", "h")
    s = s.replace(" week", "w")
    s = s.replace(" month", "m")
    s = s.replace(" year", "y")
    s = s.replace(" and", ",")
    s = s.replace("s", "")

    return s
#https://stackoverflow.com/questions/39512260/calculating-gini-coefficient-in-python-numpy
def gini(x):
    # (Warning: This is a concise implementation, but it is O(n**2)
    # in time and memory, where n = len(x).  *Don't* pass in huge
    # samples!)

    # Mean absolute difference
    mad = np.abs(np.subtract.outer(x, x)).mean()
    # Relative mean absolute difference
    rmad = mad/np.mean(x)
    # Gini coefficient
    g = 0.5 * rmad
    return g


def persist_dataframe(data_df, prefix, timestamp = ""):
    path = ApplicationPaths.evaluation_results()
    if timestamp:
        path = ApplicationPaths.evaluation_results()+timestamp+"/"
        os.makedirs(path, exist_ok=True)
    data_df.to_csv(path + prefix + ".dataframe.csv")
    print("persisted file as: " + path + prefix + ".dataframe.csv")

def persist_lat_matrix(lat_np, prefix, timestamp = ""):
    path = ApplicationPaths.evaluation_results()
    if timestamp:
        path = ApplicationPaths.evaluation_results()+timestamp+"/"
        os.makedirs(path, exist_ok=True)
    print(lat_np)
    np.save(path + prefix + ".lat_matrix", lat_np)
    np.savetxt(path + prefix + ".lat_matrix", lat_np)

    print("persisted lat matrix as: " + path + prefix + ".lat_matrix")

def get_eval_data(timestamp):
    path = str(ApplicationPaths.evaluation_results() + timestamp + os.sep)
    print(path)
    f_global = glob(path + "*global_*.dataframe.csv")[0]
    f_miner = glob(path + "*miner_*.dataframe.csv")[0]
    f_lat_mat = glob(path + "*.lat_matrix.npy")[0]
    global_df = pd.read_csv(f_global)
    miner_df =  pd.read_csv(f_miner)
    np_lat_mat = np.load(f_lat_mat)
    return (global_df, miner_df, np_lat_mat)
