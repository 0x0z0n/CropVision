import pandas as pd
features_db=pd.read_csv('new_features.csv',header=None)
features_db.tail()
import numpy as np

class_mapping = {label: idx for idx, label in enumerate(np.unique(features_db[48]))}
class_mapping
features_db[48] = features_db[48].map(class_mapping)

features_db.to_csv('new_final.csv')
X,Y = features_db.loc[:,0:47],features_db.loc[:,48]
from distutils.version import LooseVersion as Version
from sklearn import __version__ as sklearn_version
if Version(sklearn_version) < '0.18':
    from sklearn.cross_validation import train_test_split
else:
    from sklearn.model_selection import train_test_split
print(Version(sklearn_version))
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=0)
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)
from sklearn.svm import SVC
import seaborn as sns
import matplotlib.pyplot as plt
svm_model_linear = SVC(kernel = 'linear', C = 1).fit(X_train_std, y_train)
svm_predictions = svm_model_linear.predict(X_test_std)


# model accuracy for X_test
accuracy = svm_model_linear.score(X_test_std, y_test)
from sklearn.metrics import confusion_matrix


accuracy1 = round(accuracy*100,2)
from tkinter import *
root = Tk()
root.configure(background="white")
root.title("SVM Accuracy")

root.configure(background="white")
root.title("SVM Accuracy")
Label(root, text="SVM Accuracy:{}".format(accuracy1
                                          ), font=("times new roman", 15), fg="white",
          bg="#000000",
          height=2).grid(row=0, columnspan=2, sticky=N + E + W + S, padx=75, pady=15)
root.mainloop()

# creating a confusion matrix
cm = confusion_matrix(y_test, svm_predictions)
sns.heatmap(cm, cmap=plt.cm.Blues, annot=True)

plt.show()

print(cm)