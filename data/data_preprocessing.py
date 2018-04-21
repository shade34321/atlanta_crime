import numpy as np
import pandas as pd
import datetime as dt

from sklearn import preprocessing

def convert_neighborhoods( database ):
    index = 0
    list_of_neighborhoods = []

    for x in database.neighborhood:
        if not (x in list_of_neighborhoods):
            list_of_neighborhoods.append(x)
    database.neighborhood[index] = list_of_neighborhoods.index(x)
    index += 1
    return list_of_neighborhoods

def encode_classifier(data, label):
    le = preprocessing.LabelEncoder()
    le.fit(database[label].unique().tolist())
    
    return le.transform(database[label].tolist())


def create_preprocessed_csv():
    database = pd.read_csv('COBRA-YTD2017.csv')

    database.rename(columns={'UC2 Literal' : 'crime', 'MaxOfnum_victims': 'victims', 'Avg Day': 'day_of_week'}, inplace=True)
    database.head()
    database['occur_month'] = pd.to_datetime(database.occur_date).dt.month
    database['occur_day'] = pd.to_datetime(database.occur_date).dt.day
    database['rpt_month'] = pd.to_datetime(database.rpt_date).dt.month
    database['rpt_day'] = pd.to_datetime(database.rpt_date).dt.day
    database['occur_time'] = pd.to_datetime(database.occur_time).dt.hour

    database['location'] = database['location'].str.replace('\d+ ', '')

    location_le = preprocessing.LabelEncoder()
    crime_le = preprocessing.LabelEncoder()
    neighborhood_le = preprocessing.LabelEncoder()
    day_of_week_le = preprocessing.LabelEncoder()

    location_le.fit(np.array(database.location.unique()).tolist())
    crime_le.fit(np.array(database.crime.unique()).tolist())
    neighborhood_le.fit(np.array(database.neighborhood.unique()).tolist())
    day_of_week_le.fit(np.array(database.day_of_week.unique()).tolist())

    database.location = location_le.transform(np.array(database.location).tolist())
    database.crime = crime_le.transform(np.array(database.crime).tolist())
    database.neighborhood = neighborhood_le.transform(np.array(database.neighborhood).tolist())
    database.day_of_week = day_of_week_le.transform(np.array(database.day_of_week).tolist())

    database = database.join(pd.get_dummies(database.occur_time, prefix="hour"))
    database = database.join(pd.get_dummies(database.occur_day, prefix="day"))
    database = database.join(pd.get_dummies(database.occur_month, prefix="month"))
    database = database.join(pd.get_dummies(database.day_of_week, prefix="week"))

    database.victims = database.victims.fillna(0)

    df = database.filter(['occur_month', 'occur_day', 'occur_time', 'location', 
        'victims', 'day_of_week', 'crime', 'neighborhood'], axis=1)

    df = df.join(pd.get_dummies(df.occur_time, prefix="hour"))
    df = df.join(pd.get_dummies(df.occur_day, prefix="day"))
    df = df.join(pd.get_dummies(df.occur_month, prefix="month"))
    df = df.join(pd.get_dummies(df.day_of_week, prefix="week"))


    df.to_csv("preprocessed_data.csv", sep=',')

    # print(database.head())

if __name__ == '__main__':
    create_preprocessed_csv()