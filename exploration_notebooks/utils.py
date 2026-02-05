import os
from os.path import dirname, join, split
import pandas as pd
from millify import millify
from matplotlib.ticker import FuncFormatter
import matplotlib.pyplot as plt

from typing import Union

import numpy as np

ROOT_PATH = dirname(dirname(__file__))


def get_version_num(version_path:str)->float:
    
    version_folder_name = split(version_path)[1]
    
    version_num = version_folder_name.replace("_version_data", "")

    return float(version_num)

def get_iter_num(iter_path:str)->int:

    iter_folder_name = split(iter_path)[1]
    
    iter_num = iter_folder_name.replace("iter_", "")

    return int(iter_num)


def get_version_folders():
    version_folders = pd.Series(os.listdir(ROOT_PATH), name="version_folder")
    version_folders = version_folders[version_folders.str.endswith("_version_data")].sort_values()
    version_folders = version_folders.apply(lambda folder: join(ROOT_PATH, folder))

    version_folders = version_folders.to_frame()

    version_folders["version"] = version_folders['version_folder'].apply(get_version_num)

    return version_folders.sort_values("version")





def get_iter_folders():
    
    version_folders = get_version_folders()

    # Create a dataframe with all the iteration folders
    paths = []
    for version_folder in version_folders["version_folder"].values:

        iter_folders =  pd.Series(os.listdir(version_folder))

        iter_folders = iter_folders[iter_folders.str.startswith("iter_") & ~iter_folders.str.endswith("000")]
        
        iter_folders = iter_folders.apply(lambda iter_folder : join(version_folder, iter_folder))
        
        iter_folders = iter_folders.sort_values().tolist()
        
        iter_folders.pop()
        
        for iter_folder in iter_folders:
            row = dict(iter_folder = iter_folder, version_folder = version_folder)
            paths.append(row)  


    # Create Data Frame
    df_iter_path = pd.DataFrame(paths)

    df_iter_path["version"] = df_iter_path['version_folder'].apply(get_version_num)

    df_iter_path["iter"] = df_iter_path['iter_folder'].apply(get_iter_num)

    df_iter_path.sort_values(["version", 'iter'], inplace = True)

    return df_iter_path



def set_legend_outside_plot(ax):
    # Get the current handles and labels for the legend
    handles, labels = ax.get_legend_handles_labels()
    
    # Place the legend outside the plot
    ax.legend(handles=handles, labels=labels, loc='upper left', bbox_to_anchor=(1.02, 1.0), borderaxespad=0.)

def format_millify_axis(axis, which='both', precision=0):
    """
    Apply millify formatting to the tick labels of the specified axis.

    Parameters:
    axis (matplotlib.axis.Axis): The axis object to format.
    which (str): 'x', 'y', or 'both' - which axis to format.
    precision (int): Number of decimal places to display in formatted labels.
    """
    def formatter(x, _):
        return millify(x, precision)

    if which in ['x', 'both']:
        axis.get_xaxis().set_major_formatter(FuncFormatter(formatter))
    if which in ['y', 'both']:
        axis.get_yaxis().set_major_formatter(FuncFormatter(formatter))




def format_axis(axis, which='both', fmt = "{:.2f}"):
    """
    Apply millify formatting to the tick labels of the specified axis.

    Parameters:
    axis (matplotlib.axis.Axis): The axis object to format.
    which (str): 'x', 'y', or 'both' - which axis to format.
    precision (int): Number of decimal places to display in formatted labels.
    """
    def formatter(x, _):
        return fmt.format(x)

    if which in ['x', 'both']:
        axis.get_xaxis().set_major_formatter(FuncFormatter(formatter))
        
    if which in ['y', 'both']:
        axis.get_yaxis().set_major_formatter(FuncFormatter(formatter))



def autolabel(axis:Union[np.ndarray,plt.Axes], fmt = "{:.2f}", fontsize = 9):
    """Adiciona textos sobre cada barra em um gráfico de barras

    Parameters
    ----------
    ax : Union[np.ndarray,plt.Axes]
        Eixo ou eixos do matplotlib
    fmt : str
        Formato do texto apresentado
    """
    if not isinstance(axis, np.ndarray):
        axis = np.array([axis])
    
    
    for ax in axis.flatten():
        for container in ax.containers:
            ax.bar_label(container, fmt = fmt, fontsize = fontsize)




if __name__ == "__main__":
    print(get_iter_folders())
    # print(get_version_folders())

