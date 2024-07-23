import pandas as pd
import numpy as np
from scipy import stats

df = pd.read_csv('./data/dataset_analysed.csv', dtype={'StateOfBuilding' : object})

for elem in df.iterrows():
    if np.isnan(elem[1]['GardenArea']) and elem[1]['Garden'] == 0:
        df.at[elem[0], 'GardenArea'] = 0

df.drop(df[['Url', 'Country', 'MonthlyCharges','PropertyId','MunicipalityName', 'PostalCode','Garden','RefnisCode', 'Locality', 'Province', 'Region']], axis='columns',inplace=True)
df = pd.get_dummies(df, columns=['District', 'FloodingZone', 'Kitchen', 'PEB', 'StateOfBuilding' , 'SubtypeOfProperty' , 'TypeOfSale' , 'Fireplace' , 'ConstructionYear','NumberOfFacades','SwimmingPool','Terrace','TypeOfProperty','Furnished','GardenArea'], drop_first=True)
df.fillna(df.median(), inplace=True)
df = df[(np.abs(stats.zscore(df['Price']))<3)]