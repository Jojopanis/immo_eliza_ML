import pandas as pd
import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error as mae
from sklearn.ensemble import RandomForestRegressor as RFR

df = pd.read_csv('./data/dataset_analysed.csv', dtype={'StateOfBuilding' : object})

for elem in df.iterrows():
    if np.isnan(elem[1]['GardenArea']) and elem[1]['Garden'] == 0:
        df.at[elem[0], 'GardenArea'] = 0

df.drop(df[['Url', 'Country', 'MonthlyCharges','PropertyId','MunicipalityName', 'PostalCode','Garden','RefnisCode', 'Locality']], axis='columns',inplace=True)
df = pd.get_dummies(df, columns=['District', 'FloodingZone', 'Kitchen', 'PEB', 'Province', 'Region' , 'StateOfBuilding' , 'SubtypeOfProperty' , 'TypeOfSale' , 'Fireplace' , 'ConstructionYear','NumberOfFacades','SwimmingPool','Terrace','TypeOfProperty','Furnished'], drop_first=True)
df.fillna(df.median(), inplace=True)
df = df[(np.abs(stats.zscore(df['Price']))<3)]


y = df['Price']
X = df.drop('Price', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=42, test_size=0.2)

regressor = RFR(n_jobs=-1)
regressor.fit(X_train, y_train)
score = regressor.score(X_train, y_train)
mean_absolute_error = mae(y_test, regressor.predict(X_test))


print(f'The model scored {score} with a mae of {mean_absolute_error}')