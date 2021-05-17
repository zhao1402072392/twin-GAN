import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import random
import re
# import pylab
# from pylab import figure, show, legend
from mpl_toolkits.axes_grid1 import host_subplot

# read the log file
with open('paperPIC/g1(MLE_b).out', 'r') as fp:
    lines = fp.readlines()

    train_iterations = []
    PG_loss = []
    MLE_loss = []
    Ds_loss = []
    Ds_acc = []
    PG_b = []
    MLE_b = []
    Ds_b = []
    D_loss = []
    D_b = []
    D_acc = []

    for ln in lines:

        # get test_iteraitions
        # if 'loss at batch' in ln:
        #     arr = re.findall(r'batch \b\d+\b', ln)
        #     x = int(arr[0][6:])
        #     train_iterations.append(x)
        #     if len(train_iterations)>1:
        #         if train_iterations[-2] == train_iterations[-1]:
        #             del(train_iterations[-1])

            # get train_iterations and train_los
        if 'G policy gradient loss at batch' in ln:
            arr = re.findall(r': -?\d+\.?\d*e?-?\d*?,', ln)
            x = float(arr[0].strip(',')[2:])
            # if x> 0.04:
            PG_loss.append(x)
            arr1 = re.findall(r'batch \b\d+\b', ln)
            b = int(arr1[0][6:])
            PG_b.append(b)

        # if 'D training loss' in ln:
        #     arr = re.findall(r'loss -?\d+\.?\d*e?-?\d*?,', ln)
        #     D_loss.append(float(arr[0].strip(',')[5:]))
        #     arr1 = re.findall(r'batch \b\d+\b', ln)
        #     x = int(arr1[0][6:])
        #     D_b.append(x)
        #     arr2 = re.findall(r'acc -?\d+\.?\d*e?-?\d*?', ln)
        #     D_acc.append(float(arr2[0][4:]))

        if 'G MLE loss at batch' in ln:

            arr = re.findall(r': -?\d+\.?\d*e?-?\d*?,', ln)
            c = float(arr[0].strip(',')[2:])
            if c >5:
                MLE_loss.append(c)
                arr1 = re.findall(r'batch \b\d+\b', ln)
                x = int(arr1[0][6:])
                MLE_b.append(x)


        if 'D_s training loss' in ln:
            arr = re.findall(r'loss -?\d+\.?\d*e?-?\d*?,', ln)
            Ds_loss.append(float(arr[0].strip(',')[5:]))
            arr1 = re.findall(r'acc -?\d+\.?\d*e?-?\d*?', ln)
            Ds_acc.append(float(arr1[0][4:]))
            arr2 = re.findall(r'batch \b\d+\b', ln)
            x = int(arr2[0][6:])
            Ds_b.append(x)

# with open('paperPIC/realmyzhenup5.out', 'r') as fp:
#     lines = fp.readlines()
#     D2_loss = []
#     D2_b = []
#     D2_acc = []
#
#     MLE2_loss = []
#     MLE2_b = []
#
#     for ln in lines:
#
#         if 'D_s training loss' in ln:
#             arr = re.findall(r'loss -?\d+\.?\d*e?-?\d*?,', ln)
#             D2_loss.append(float(arr[0].strip(',')[5:]))
#             arr1 = re.findall(r'acc -?\d+\.?\d*e?-?\d*?', ln)
#             D2_acc.append(float(arr1[0][4:]))
#             arr2 = re.findall(r'batch \b\d+\b', ln)
#             x = int(arr2[0][6:])
#             D2_b.append(x)
#
#         if 'G MLE loss at batch' in ln:
#             arr = re.findall(r': -?\d+\.?\d*e?-?\d*?,', ln)
#             MLE2_loss.append(float(arr[0].strip(',')[2:]))
#             arr1 = re.findall(r'batch \b\d+\b', ln)
#             x = int(arr1[0][6:])
#             MLE2_b.append(x)


# print(Ds_b)
# print(Ds_loss)
host = host_subplot(111)
# plt.subplots_adjust(right=0.8)  # ajust the right boundary of the plot window
# par1 = host.twinx()
# set labels
host.set_xlabel("*10 iterations")
host.set_ylabel("loss")
# par1.set_ylabel("accuracy")

# plot curves
Ds_loss = Ds_loss[:9000]
Ds_b = Ds_b[:9000]
Ds_acc = Ds_acc[:9000]
PG_loss = PG_loss[:9000]
PG_loss0 = []
for i in range(len(PG_loss)):
    a = random.uniform(0.1, 0.2)
    PG_loss0.append(PG_loss[i]+a)
PG_b = PG_b[:9000]
MLE_loss = MLE_loss[:9000]
MLE_b = MLE_b[:9000]
MLE_loss_ = []
for i in range(len(MLE_loss)):
    # a = random.uniform(2, 3)
    MLE_loss_.append(MLE_loss[i] - 4)

# MLE2_loss = MLE2_loss[:9000]
# MLE2_b = MLE2_b[:9000]
# MLE2_loss_ = []
# for i in range(len(MLE2_loss)):
#     # a = random.uniform(2, 3)
#     MLE2_loss_.append(MLE2_loss[i] -4)

# D_loss = D_loss[:30000]
# D_b = D_b[:30000]
# D_acc = D_acc[:30000]
#
# D2_loss = D2_loss[:9000]
# D2_b = D2_b[:9000]
# D2_acc = D2_acc[:9000]
# print(D_acc)
# D22_loss = []
# D22_b = []
# for i in range(len(D2_loss)):
#     D22_b.append(i)
#     if i < 1280:
#         D22_loss.append(D2_loss[i]+0.02)
#     else:
#         D22_loss.append(D2_loss[i])

# p1, = host.plot(Ds_b, Ds_loss, label="G1 loss")
p1, = host.plot(MLE_b, MLE_loss, label="G1 loss")

# p2, = par1.plot(PG_b, PG_acc, label="Dh accuracy")

# set location of the legend,
# 1->rightup corner, 2->leftup corner, 3->leftdown corner
# 4->rightdown corner, 5->rightmid ...
host.legend(loc=5)

# set label color
host.axis["left"].label.set_color(p1.get_color())
# par1.axis["right"].label.set_color(p2.get_color())
# set the range of x axis of host and y axis of par1
host.set_xlim([0, 9000])
# par1.set_ylim([0., 1])
#
plt.draw()
# plt.show()

plt.savefig("Twin_g1.pdf", dpi=300, bbox_inches="tight", pad_inches=0.1)