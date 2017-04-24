# Use generated proximity matrix generated to predict rating

import sys
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
import math


distanceMatrix = pd.read_csv("../Data/DistanceMatrix.csv")
# For each userId we have the IDs of the 20 closest neighbors

cols = range(1,len(distanceMatrix.columns))
distanceMatrix = distanceMatrix[cols]
print distanceMatrix.head()

Data = pd.read_csv("../Data/train.csv")

with open('../Data/trainIndex.txt') as f:
    content = f.readlines()
ind = [int(x) for x in content] 

Data = Data.iloc[ind]

RATE_INTERVAL = 10
Data['Rating'] = (Data['Rating'].astype(int)/RATE_INTERVAL).astype(int)

trainData = Data

print trainData.head()
testData = trainData.copy()  #<----_Change

trainData = Data.sample(frac=0.7).copy()
testData = Data.drop(trainData.index).copy()

testData['predictedRating'] = int(0)
# Nearest users ... average rating given to track
print testData['Rating'].value_counts()

for (index, row) in testData.iterrows():
	print index
	user = row['User']

	neighbors = distanceMatrix.loc[distanceMatrix['user']==user]

	neighbors = trainData[trainData['User'].isin(neighbors.values[0])]
	
	
	if neighbors[neighbors['Track']==row['Track']].shape[0]!=0:
		meanVal = neighbors[neighbors['Track']==row['Track']]['Rating'].mean()
	elif neighbors[neighbors['Artist']==row['Artist']].shape[0]!=0:
		meanVal = neighbors[neighbors['Artist']==row['Artist']]['Rating'].mean()
	else:
		meanVal = neighbors['Rating'].mean()
	testData.loc[index,'predictedRating'] = int(round(meanVal))



print accuracy_score(testData['Rating'], testData['predictedRating'])

print confusion_matrix(testData['Rating'], testData['predictedRating'])


