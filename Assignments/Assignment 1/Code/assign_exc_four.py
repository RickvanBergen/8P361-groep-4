# -*- coding: utf-8 -*-
"""
Assignment 1
Exercise 4, code

"""
import numpy as np
import os
import glob
from PIL import Image
import random
import tkinter as tk
from tkinter import filedialog

def folder_path():
    #for class 0, take path "./train/0"
    #for class 1, take path "./train/1"
    root= tk.Tk()
    root.directory = filedialog.askdirectory()
    root.withdraw()
    return root.directory

def displayimages(image_path):
    #displays 4 random images from image path
    extension = "*.jpg"
    directory = os.path.join(image_path, extension)
    files = glob.glob(directory)
    rndmlist=random.sample(files, 4)

    for file in rndmlist:
        im = Image.open(file)
        im.show()

displayimages(folder_path())

