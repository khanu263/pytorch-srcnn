# plot.py
# by Umair Khan
# CS 410 - Spring 2020

# Plot validation PSNR versus time for all zoom levels.
# (This is hardcoded, probably not very useful.)

# Imports
import sys
import os
import time
import math
import argparse

# Import matplotlib and seaborn (X workaround) and style
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
sns.set_context("talk")
matplotlib.rcParams["font.family"] = "Public Sans"
matplotlib.rcParams["font.weight"] = "semibold"
matplotlib.rcParams["axes.titleweight"] = "bold"
matplotlib.rcParams["axes.labelweight"] = "medium"
matplotlib.rcParams["axes.titlesize"] = "medium"
matplotlib.rcParams["axes.labelsize"] = "small"
matplotlib.rcParams["axes.labelpad"] = 20.0
matplotlib.rcParams['xtick.labelsize'] = "small"
matplotlib.rcParams['ytick.labelsize'] = "small"
matplotlib.rcParams['text.color'] = "#333333"
matplotlib.rcParams['axes.labelcolor'] = "#333333"
matplotlib.rcParams['xtick.color'] = "#333333"
matplotlib.rcParams['ytick.color'] = "#333333"
matplotlib.rcParams["axes.facecolor"] = "white"
matplotlib.rcParams["axes.edgecolor"] = "#333333"
matplotlib.rcParams["lines.linewidth"] = 2.5
matplotlib.rcParams["axes.spines.top"] = False
matplotlib.rcParams["axes.spines.right"] = False
matplotlib.rcParams["legend.fontsize"] = 15

# Colors to cycle through
COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan"]

# Load in all data
psnrs = {}
for i in range(2, 9):
    lines = open("7000_z{}.txt".format(i), "r").read().splitlines()
    psnrs[i] = [float(x) for x in lines[1].split("\t")[1:]]
print("Loaded data.")

# Plot PSNR
x_ax = list(range(1, 7001))
plt.figure(figsize = (10, 6))
plt.xlabel("Epoch")
plt.ylabel("Average PSNR (dB)")
for z in list(psnrs.keys()):
    plt.plot(x_ax, psnrs[z], COLORS[(z - 2) % len(COLORS)], label = "z{}".format(z))
plt.legend(loc = "best")
plt.tight_layout()
plt.savefig("../figs/all_val.png")
print("Saved plot.")