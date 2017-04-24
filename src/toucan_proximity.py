# Create Users Proximity Matrix

import sys
import pandas as pd 
from sklearn.metrics.pairwise import pairwise_distances

users = pd.read_csv('../Data/UserDataProcessed.csv')

cols = users.columns[1:]

distanceMatrix = []
users.index = users['user']
it=0
for (index, row) in users.iterrows():
	print it
	n=21
	d = {}
	distances = (users[cols] != row[cols]).sum(axis=1).sort_values()[:n].index
	for i in range(n):
		d[i] = distances[i]
	d['user'] = row['user']
	distanceMatrix.append(d)
	it=it+1

pd.DataFrame(distanceMatrix).to_csv("../Data/DistanceMatrix.csv", index=False)

