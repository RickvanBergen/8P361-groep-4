import numpy as np
import pandas as pd
import os
import tkinter as tk
from tkinter import filedialog

def unique(list1):
    # intilize a null list
    unique_list = []

    # traverse for all elements
    for x in list1:
        # check if exists in unique_list or not
        if x not in unique_list:
            unique_list.append(x)
            # print list
    return unique_list

def folder_path():
    root= tk.Tk()
    root.directory = filedialog.askdirectory()
    root.withdraw()
    return root.directory

def file_path():
    root = tk.Tk()
    root.withdraw()
    path = filedialog.askopenfilename()
    return path