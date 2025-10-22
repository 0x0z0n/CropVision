TRAIN_PATH = "/train"
VAL_PATH = "/val"

from numpy import *
from tkinter import *
root = Tk()
root.configure(background="white")
import os
def datapreprocessing():
    os.system("py fe.py")
    print("Data Preprocessing Done!")


def svm():
    os.system("py svm.py")

def dtree():
    os.system("py decisiontree.py")


def plotloss():
    os.system("py comparison.py")


def lr():
    os.system("py logisticregression.py")

def cnn():
    os.system("py cnn.py")


def function6():
    root.destroy()
def appopen():
    os.system("py app.py")
# stting title for the window
root.title("Leaf Disease Prediction System")

# creating a text label
Label(root, text="Leaf Disease Prediction System", font=("times new roman", 20), fg="white", bg="#795548",
      height=2).grid(row=0, rowspan=2, columnspan=2, sticky=N + E + W + S, padx=5, pady=5)

# creating first button
Button(root, text="Data Preprocessing", font=('times new roman', 20), bg="#bcaaa4", fg="#3e2723", command=datapreprocessing).grid(
    row=5, columnspan=2, sticky=N + E + W + S, padx=5, pady=5)
Button(root, text="Decision Tree ", font=('times new roman', 20), bg="#bcaaa4", fg="#3e2723", command=dtree).grid(
    row=6, columnspan=2, sticky=N + E + W + S, padx=5, pady=5)
Button(root, text="SVM", font=('times new roman', 20), bg="#bcaaa4", fg="#3e2723", command=svm).grid(
    row=7, columnspan=2, sticky=N + E + W + S, padx=5, pady=5)
# creating second button
Button(root, text="Logistic Regression", font=('times new roman', 20), bg="#bcaaa4", fg="#3e2723", command=lr).grid(
    row=8, columnspan=2, sticky=N + E + W + S, padx=5, pady=5)
Button(root, text="Model Training using CNN", font=('times new roman', 20), bg="#bcaaa4", fg="#3e2723", command=cnn).grid(
    row=9, columnspan=2, sticky=N + E + W + S, padx=5, pady=5)
Button(root, text="Accuracy Comparison", font=('times new roman', 20), bg="#bcaaa4", fg="#3e2723", command=plotloss).grid(
    row=10, columnspan=2, sticky=N + E + W + S, padx=5, pady=5)
Button(root, text="Leaf Disease Prediction"
                  "", font=('times new roman', 20), bg="#bcaaa4", fg="#3e2723", command=appopen).grid(
    row=11, columnspan=2, sticky=N + E + W + S, padx=5, pady=5)
Button(root, text="Exit", font=('times new roman', 20), bg="#795548", fg="white", command=function6).grid(row=12,
                                                                                                         columnspan=2,
                                                                                                         sticky=N + E + W + S,
                                                                                                         padx=5, pady=5)

root.mainloop()
