import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
cols=["fLenght","fWidth" , "fSize","fConc","fAsym","fM3long","fM3trans","fAlpha","fDist","class"]
df=pd.read_csv("magic04.data",names=cols)
df["class"]=(df["class"]=="g").astype(int)

for label in df.columns[:-1]:
  plt.hist(df[df["class"]==1][label],color="blue",label="gamma", alpha=0.7, density=True)
  plt.hist(df[df["class"]==0][label],color="red",label="hardon", alpha=0.7, density=True)
  plt.xlabel(label)
  plt.ylabel("probability")
  plt.legend()
  plt.show()
  


train, valid, test = np.split(df.sample(frac=1).reset_index(drop=True), [int(0.6*len(df)), int(0.8*len(df))])



def scale_dataset(dataframe , oversampler=False):
  x=dataframe[dataframe.columns[:-1]].values
  y=dataframe[dataframe.columns[-1]].values

  scaler=StandardScaler()
  x=scaler.fit_transform(x)
  if oversampler:
    ros=RandomOverSampler()
    x,y=ros.fit_resample(x,y)

  data=np.hstack((x,np.reshape(y,(-1,1))))
  return data , x ,y



train , xtrain , ytrain=scale_dataset(train,oversampler=True)
valid , xvalid , yvalid=scale_dataset(valid,oversampler=False)
test , xtest , ytest=scale_dataset(test,oversampler=False)

# K N N (k nearest neighbors)

from sklearn.neighbors import KNeighborsClassifier

knn_model=KNeighborsClassifier(n_neighbors=1)
knn_model.fit(xtrain,ytrain)

ypred=knn_model.predict(xtest)
print(ypred)
print(classification_report(ytest,ypred))
