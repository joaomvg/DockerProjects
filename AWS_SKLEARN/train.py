from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import pickle

iris=load_iris()

X=iris['data']
y=iris['target']

if __name__=='__main__':
    lgr=LogisticRegression()
    lgr.fit(X,y)
    acc_train=accuracy_score(lgr.predict(X),y)
    print('Score training: ',acc_train)
        
    pickle.dump(lgr,open('lgr.pkl','wb'))



