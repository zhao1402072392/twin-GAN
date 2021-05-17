import numpy as np
import matplotlib.pyplot as plt
import pylab as pl
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import re


with open('paperPIC/original.out', 'r') as fp:
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
            PG_loss.append(float(arr[0].strip(',')[2:]))
            arr1 = re.findall(r'batch \b\d+\b', ln)
            x = int(arr1[0][6:])
            PG_b.append(x)

        if 'D training loss' in ln:
            arr = re.findall(r'loss -?\d+\.?\d*e?-?\d*?,', ln)
            D_loss.append(float(arr[0].strip(',')[5:]))
            arr1 = re.findall(r'batch \b\d+\b', ln)
            x = int(arr1[0][6:])
            D_b.append(x)
            arr2 = re.findall(r'acc -?\d+\.?\d*e?-?\d*?', ln)
            D_acc.append(float(arr2[0][4:]))

        if 'G MLE loss at batch' in ln:
            arr = re.findall(r': -?\d+\.?\d*e?-?\d*?,', ln)
            MLE_loss.append(float(arr[0].strip(',')[2:]))
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

PG_loss = PG_loss[:30000]
PG_b = PG_b[:30000]
MLE_loss = MLE_loss[:30000]
MLE_b = MLE_b[:30000]
x = PG_b
y = PG_loss
x1 = MLE_b
y1 = MLE_loss
fig = plt.figure(figsize = (7,5))    #figsize是图片的大小`
ax1 = fig.add_subplot(1, 1, 1) # ax1是子图的名字`

plt.plot(x,y,'g-',label=u'Dense_Unet(block layer=5)')
# ‘'g‘'代表“green”,表示画出的曲线是绿色，“-”代表画的曲线是实线，可自行选择，label代表的是图例的名称，一般要在名称前面加一个u，如果名称是中文，会显示不出来，目前还不知道怎么解决。
p2 = plt.plot(x1, y1,'r-', label = u'RCSCA_Net')
pl.legend()
#显示图例
# p3 = pl.plot(x2,y2, 'b-', label = u'SCRCA_Net')
plt.legend()
plt.xlabel(u'iters')
plt.ylabel(u'loss')
plt.title('Compare loss for different models in training')