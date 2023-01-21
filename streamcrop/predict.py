import pandas as pd
import numpy as np
import joblib as jb
import pickle
from sklearn.model_selection import train_test_split
df=pd.read_csv("Crop_recommendation.csv")
x=df.drop('label',axis=1)
y=df['label']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3,shuffle = True, random_state = 0)
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(x_train, y_train)
y_pred=model.predict(x_test)
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_pred, y_test)
print('LightGBM Model accuracy score: {0:0.4f}'.format(accuracy_score(y_test, y_pred)))
pickle_out=open("model.pkl","wb")
pickle.dump(model,pickle_out)
pickle_out.close()
