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
    return walltime,step,value
    
def files():
    value=[]
    types=['-','--','*']
    path_acc = ut.folder_path()
    files_acc = os.listdir(path_acc)
    path_loss = ut.folder_path()
    files_loss = os.listdir(path_loss)
    i=0

    for file in files_acc:
        # num=str(i+1)
        # file=input("Enter filename of CSV file "+num+ " with extension: ")

        t,s,v= readcsv(os.path.join(path_acc,file))

        t=t[1:]
        s=s[1:]
        v=v[1:]

        #Rounding necessary because of a bug in plotting
        for val in range(len(v)):
            v[val]=round(float(v[val]),9)
        
        label = file.split('-')[1]
        plt.figure(1)
        plt.plot(s,v,types[i]+'k',label=label)
        value.append(v)
        i+=1
    plt.legend(fontsize=20)
    plt.tick_params(labelsize=20)
    plt.xlabel("Steps", fontdict={'fontsize': 25})
    plt.ylabel("Accuracy", fontdict={'fontsize':25})
    plt.title("Training and validation accuracy of " + file.split('-')[0], fontdict={'fontsize': 30})
    plt.grid() 
    plt.show()
    
    i=0
    for file in files_loss:
        # num=str(i+1)
        # file=input("Enter filename of CSV file "+num+ " with extension: ")

        t,s,v= readcsv(os.path.join(path_loss,file))

        t=t[1:]
        s=s[1:]
        v=v[1:]

        #Rounding necessary because of a bug in plotting
        for val in range(len(v)):
            v[val]=round(float(v[val]),9)
        
        label = file.split('-')[1]
        plt.figure(2)
        plt.plot(s,v,types[i]+'k',label=label)
        value.append(v)
        i+=1
    
    plt.legend(fontsize=20)
    plt.tick_params(labelsize=20)
    plt.xlabel("Steps", fontdict={'fontsize': 25})
    plt.ylabel("Loss", fontdict={'fontsize': 25})
    plt.title("Training and validation loss of " + file.split('-')[0], fontdict={'fontsize': 30})
    plt.grid() 
    plt.show()

files()
##