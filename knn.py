# loading libraries
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import cross_val_score
from collections import Counter
from sklearn import preprocessing


def loadData(v):
	names = ['rpt_month', 'rpt_day', 'occur_month', 'occur_day', 'occur_time', 'location', 'victims', 'day_of_week', 'neighborhood', 'crime']

	data = pd.read_csv('data/preprocessed_data.csv',skiprows=[0], names=names)
	location_le = preprocessing.LabelEncoder()
	crime_le = preprocessing.LabelEncoder()
	neighborhood_le = preprocessing.LabelEncoder()
	day_of_week_le = preprocessing.LabelEncoder()

	location_le.fit(data.location.unique().tolist())
	crime_le.fit(data.crime.unique().tolist())
	neighborhood_le.fit(data.neighborhood.unique().tolist())
	day_of_week_le.fit(data.day_of_week.unique().tolist())

	data.location = location_le.transform(data.location.tolist())
	data.crime = crime_le.transform(data.crime.tolist())
	data.neighborhood = neighborhood_le.transform(data.neighborhood.tolist())
	data.day_of_week = day_of_week_le.transform(data.day_of_week.tolist())

	#data.head()
	if v == 0:
		X = data.drop(['crime'], axis=1)
		y = data['crime']
	else:
		X = np.array(data.drop(['crime'], axis=1))
		y = np.array(data['crime'])

	# split into train and test
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
	return X_train, X_test, y_train, y_test

def train(X_train, y_train):
	# do nothing 
	return

def predict(X_train, y_train, x_test, k):
	# create list for distances and targets
	distances = []
	targets = []

	for i in range(len(X_train)):
		# first we compute the euclidean distance
		distance = np.sqrt(np.sum(np.square(x_test - X_train[i, :])))
		# add it to list of distances
		distances.append([distance, i])

	# sort the list
	distances = sorted(distances)

	# make a list of the k neighbors' targets
	for i in range(k):
		index = distances[i][1]
		#print(y_train[index])
		targets.append(y_train[index])

	# return most common target
	return Counter(targets).most_common(1)[0][0]

def kNearestNeighbor(X_train, y_train, X_test, predictions, k):
	# check if k is not larger than n
	if k > len(X_train):
		raise ValueError
		
	# train on the input data
	train(X_train, y_train)

	# predict for each testing observation
	for i in range(len(X_test)):
		predictions.append(predict(X_train, y_train, X_test[i, :], k))

def ParameterTuningCrossValidation(X_train, y_train):
	# creating odd list of K for KNN
	myList = list(range(0,50))
	neighbors = list(filter(lambda x: x % 2 != 0, myList))

	# empty list that will hold cv scores
	cv_scores = []

	# perform 10-fold cross validation
	for k in neighbors:
		knn = KNeighborsClassifier(n_neighbors=k)
		scores = cross_val_score(knn, X_train, y_train, cv=10, scoring='accuracy')
		cv_scores.append(scores.mean())

	# changing to misclassification error
	MSE = [1 - x for x in cv_scores]

	# determining best k
	optimal_k = neighbors[MSE.index(min(MSE))]
	print('\nThe optimal number of neighbors is %d.' % optimal_k)
	return optimal_k
	# plot misclassification error vs k 
	#plt.plot(neighbors, MSE)
	#plt.xlabel('Number of Neighbors K')
	#plt.ylabel('Misclassification Error')
	#plt.show()

def libraryKNN(X_train, X_test, y_train, y_test,optimal_k):

	knn = KNeighborsClassifier(n_neighbors=optimal_k)

	# fitting the model
	knn.fit(X_train, y_train)

	# predict the response
	pred = knn.predict(X_test)

	# evaluate accuracy
	acc = accuracy_score(y_test, pred) * 100
	print('\nThe accuracy of the knn classifier for k = %d is %d%%' % (optimal_k, acc))

def manualKNN(X_train, X_test, y_train, y_test,optimal_k):
	predictions = []
	try:
		kNearestNeighbor(X_train, y_train, X_test, predictions, 7)
		predictions = np.asarray(predictions)

		# evaluating accuracy
		accuracy = accuracy_score(y_test, predictions) * 100
		print('\nThe accuracy of OUR classifier is %d%%' % accuracy)

	except ValueError:
		print('Can\'t have more neighbors than training samples!!')


def main():
	
	print("\n\nLoading train and test data\n")
	X_train, X_test, y_train, y_test = loadData(0)
	X_train = X_train.fillna(-1)
	y_train = y_train.fillna(-1)
	X_test = X_test.fillna(-1)
	y_test = y_test.fillna(-1)

	print("Finding Optimal K\n")	
	optimal_k = ParameterTuningCrossValidation(X_train, y_train)

	print("Preforming KNN using skleran Libraries\n")
	libraryKNN(X_train, X_test, y_train, y_test,optimal_k)

	print("\n\nLoading train and test data\n")
	X_train, X_test, y_train, y_test = loadData(1)

	print("Performing KNN using manual code\n")
	manualKNN(X_train, X_test, y_train, y_test,optimal_k)
	

if __name__== "__main__":
  main()



