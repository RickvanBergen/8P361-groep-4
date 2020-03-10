# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 15:51:22 2020

@author: 20160824
"""

import csv
import matplotlib.pyplot as plt
import Util as ut
import os

def readcsv(read_filename):
    with open(read_filename) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        walltime = []
        step= []
        value = []
        for row in readCSV:
            time = row[0]
            stp= row[1]
            val = row[2]
            walltime.append(time)
            step.append(stp)
            value.append(val)
    #step correction
    step.pop()
    step.append(9)
    return walltime,step,value
    
def files():
    value=[]
    colours=['r','g','b','k','y','m','c','orange','gray']
    path = ut.folder_path()
    files = os.listdir(path)
    i=0
    for file in files:
        # num=str(i+1)
        # file=input("Enter filename of CSV file "+num+ " with extension: ")

        t,s,v= readcsv(os.path.join(path,file))

        t=t[1:]
        s=s[1:]
        v=v[1:]


        #Rounding necessary because of a bug in plotting
        for val in range(len(v)):
            v[val]=round(float(v[val]),9)
        plt.plot(s,v,colours[i],label=file.split('-')[1])
        value.append(v)
        i+=1
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("Accuracy")
    plt.grid()
    # plt.show(block = False)
    save = input('Save fig? [y/n]: ')
    if save == 'y':
        pathlist = path.split('/')
        savename = '_'.join(pathlist[-3:])+'.png'
        saveloc = os.path.join('/'.join(pathlist[:-4]),'Figures')
        plt.savefig(os.path.join(saveloc,savename))

        # plt.savefig(os)
    plt.show()

##Fill in number of files to be analysed (maximum of 7)
files()
##