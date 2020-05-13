import pandas as pd
weather=pd.read_csv('weatherAUS.csv',encoding="shift-jis")
weather=weather.drop(['Date','Location','Evaporation','Sunshine','WindGustDir','WindDir9am','WindDir3pm','WindSpeed9am','WindSpeed3pm','Humidity3pm','Cloud3pm','Temp3pm','RISK_MM'],axis=1)
mean=round(weather['MinTemp'].mean(),2)
mean_mt=round(weather['MaxTemp'].mean(),2)
mean_rf=round(weather['Rainfall'].mean(),2) 
mean_t9=round(weather['Temp9am'].mean(),2)
https://www.kaggle.com/jsphyg/weather-dataset-rattle-packagemean_h9=round(weather['Humidity9am'].mean(),2)
mean_p9=round(weather['Pressure9am'].mean(),2)
mean_p3=round(weather['Pressure3pm'].mean(),2)
mean_c9=round(weather['Cloud9am'].mean(),2)
mean_wg=round(weather['WindGustSpeed'].mean(),2)
weather['MinTemp'].fillna(mean,inplace=True)
weather['MaxTemp'].fillna(mean_mt,inplace=True)
weather['Rainfall'].fillna(mean_rf,inplace=True)
weather['Temp9am'].fillna(mean_t9,inplace=True)
weather['Humidity9am'].fillna(mean_h9,inplace=True)
weather['Pressure9am'].fillna(mean_p9,inplace=True)
weather['Pressure3pm'].fillna(mean_p3,inplace=True)
weather['WindGustSpeed'].fillna(mean_wg,inplace=True)
weather['Cloud9am'].fillna(mean_c9,inplace=True)
weather.fillna("",inplace=True)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for i in weather.columns.values.tolist():
 if (i=='MinTemp' or i=='MaxTemp' or i=='Rainfall'):
  pass
 else:
  weather[i] = le.fit_transform(weather[i])

from sklearn.model_selection import train_test_split
weather_target = weather['RainTomorrow']
weather_data=weather.drop(['RainTomorrow'],axis=1)
yX=weather_target
yX=pd.concat([yX,weather_data],axis=1)
yX.to_csv('temp_weather.csv',encoding='utf-8')

from imblearn.over_sampling import SMOTE
smt = SMOTE(random_state=90)
X_train,X_test,y_train,y_test=train_test_split(weather_data,weather_target,test_size=0.2,random_state=54,shuffle=True)
X_train,y_train=smt.fit_resample(X_train,y_train)
from sklearn.ensemble import RandomForestClassifier
clf=RandomForestClassifier(n_estimators=382, max_depth=None,min_samples_split=2,random_state=8)
clf.fit(X_train,y_train)
print(clf.score(X_test,y_test))
dic=dict(zip(weather_data.columns,clf.feature_importances_))
for item in sorted(dic.items(), key=lambda x: x[1], reverse=True):
    print(item[0],round(item[1],4))
