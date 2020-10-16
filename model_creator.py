from  sklearn import  datasets
from sklearn.model_selection import train_test_split
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


iris=datasets.load_iris()
X=iris.data
Y=iris.target



X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=.3)




model_lr=LogisticRegression()
model_lr.fit(X_train, Y_train)

y_pred=model_lr.predict(X_test)
acc=accuracy_score(y_pred,Y_test)

with open('./model.pkl','wb') as model_p:
    pickle.dump(model_lr,model_p)