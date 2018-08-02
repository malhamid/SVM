#$%matplotlib inline

#The following liberaries are used for analysis.
import pandas as pd 
import numpy as np 
from sklearn import svm 

#The following libraries are used to visualzie results. 
import matplotlib.pyplot as plt 
import seaborn as sns; sns.set(font_scale=1.2)

print ("Welcome to this tutorial, you will learn how to classify data using SVM")

#First step is loading the data from outsource file. CSV in this case. 
dataset = pd.read_csv('data.csv')
dataset.shape 
#Step2: printing the content of the CSV file. 
print(dataset.head())
#You can choose two factors to classify the data on. Factor 3 would be the x-axis and Factor 4 is the y-axis
sns.lmplot('Factor3', 'Factor4', data=dataset, hue='Type', palette='Set1', fit_reg=False, scatter_kws={"s":70})
#Step 3: displaying the data on two dicmentional graph
plt.show()

#Step 4: starting the clasisfication process. 
clf = dataset[['Factor3','Factor4']].values
type_label = np.where(dataset['Type']=='Car',0,1)
#SVC with linear kernel
model = svm.SVC(kernel='linear')
model.fit(clf,type_label)
#This step is for viualizing the linespace of the classifier, is not necessary. You can go ahead and do the prediction. 
w = model.coef_[0]
print(w)
a= -w[0]/w[1]
xx=np.linspace(5,30)
yy=a* xx - model.intercept_[0]/w[1]

b=model.support_vectors_[0]
yy_down=a*xx+(b[1]-a*b[0])
b = model.support_vectors_[-1]
yy_up=a*xx+(b[1]-a*b[0])
# Final step displaying the results. 
sns.lmplot('Factor3', 'Factor4', data=dataset, hue='Type', palette='Set1', fit_reg=False, scatter_kws={"s":70})
plt.plot(xx,yy,linewidth=2,color='black')
plt.plot(xx,yy_down,'k--')
plt.plot(xx,yy_up,'k--')
plt.scatter(model.support_vectors_[:,0],model.support_vectors_[:,1],s=80,facecolors='none')
plt.show()