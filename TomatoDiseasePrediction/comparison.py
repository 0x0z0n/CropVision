from matplotlib.pyplot import *
y1 = [82.4,70.5,78.3,87]
x = ["SVM","DT","LR","CNN"]
f = figure()
width = 0.4
fig = f.add_axes([0.1,0.1,0.8,0.8])
fig.bar([0,1,2,3],y1,label="Accuracy Comparison",width=width)
fig.legend()
xticks([0,1,2,3],x)
show()