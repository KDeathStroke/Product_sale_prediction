import pandas as pd
import numpy as np

dataset=pd.read_csv('data.csv')
y=dataset['Purchased'].values
dataset=dataset.drop(['Purchased','User ID'],axis=1).values



from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
dataset[:, 0]=le.fit_transform(dataset[:, 0])

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(dataset, y, test_size = 0.2, random_state = 0)



from sklearn import tree
model=tree.DecisionTreeClassifier(criterion='entropy')
model.fit(X_train,y_train)


prediction=model.predict(X_test)

error=np.mean(y_test!=prediction)
error

from sklearn.metrics import accuracy_score
testing_accuracy=accuracy_score(y_test,prediction)
testing_accuracy