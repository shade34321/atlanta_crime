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

    df = database.filter(['rpt_month', 'rpt_day', 'occur_month', 'occur_day', 'occur_time', 'location', 'victims', 'day_of_week', 'crime', 'neighborhood'], axis=1)

    df.to_csv("preprocessed_data.csv", sep=',')