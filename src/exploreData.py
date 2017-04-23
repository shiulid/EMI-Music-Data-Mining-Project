from __future__ import division
import pandas as pd 
import sys
import numpy as np 
import matplotlib.pyplot as plt

trainData = pd.read_csv('../Data/train.csv')
trainData.columns = [c.lower() for c in trainData.columns]

rating = {}
for i in range(25):
	rating[i]=0
for i in range(25,50):
	rating[i]=1
for i in range(50,75):
	rating[i]=2
for i in range(75,101):
	rating[i]=3

trainData['rating'] = trainData['rating'].map(rating)

testData = pd.read_csv('../Data/test.csv')

testData.columns = [c.lower() for c in testData.columns]

# print "Training Data"
# print trainData.shape
# print trainData.head()

# print "Testing Data"
# print testData.shape
# print testData.head()

# artists_mean_benchmark = pd.read_csv('Data/artists_mean_benchmark.csv', header=None)
# global_mean_benchmark = pd.read_csv('Data/global_mean_benchmark.csv', header=None)
# tracks_mean_benchmark = pd.read_csv('Data/tracks_mean_benchmark.csv', header=None)
# users_mean_benchmark = pd.read_csv('Data/users_mean_benchmark.csv', header=None)

# print "Artists Mean Benchmark"
# print artists_mean_benchmark.shape
# print artists_mean_benchmark.head()

# print "Global Mean Benchmark"
# print global_mean_benchmark.shape
# print global_mean_benchmark.head()

# print "Tracks Mean Benchmark"
# print tracks_mean_benchmark.shape
# print tracks_mean_benchmark.head()

# print "Users Mean Benchmark"
# print users_mean_benchmark.shape
# print users_mean_benchmark.head()

"""
Preprocessing Words file
"""

words = pd.read_csv('../Data/words.csv')
words.columns = [n.lower() for n in words.columns]
words = words.iloc[:, :-1]
print len(words.columns)
words = words.fillna(value=0)

print "Words"
# print (words.columns)
# print words.describe()

words.loc[words['heard_of']=='Never heard of','own_artist_music']='Own none of their music'
heardOfMap = {'Heard of':1, 'Heard of and listened to music EVER':1, 'Heard of and listened to music RECENTLY':1, 'Ever heard music by':1, 'Listened to recently':1  , 'Ever heard of':1, 'Never heard of':0, 0:0}
ownMusicMap = {'Own a lot of their music':2,'Own all or most of their music':2, 'Own a little of their music':1, 'Don\xcdt know':1,'Dont know':1, 'don`t know':1, 'Don\xd5t know':1, 'Own none of their music':0, 0:0}

with open('../Data/positiveWords.txt') as f:
    content = f.readlines()
positiveWords = [x.strip().lower() for x in content] 

with open('../Data/negativeWords.txt') as f:
    content = f.readlines()
negativeWords = [x.strip().lower() for x in content] 

with open('../Data/neutralWords.txt') as f:
    content = f.readlines()
neutralWords = [x.strip().lower() for x in content] 

# Create new db, group heard of yes/no, own artist music, like rating, words list
# group by artist
words['heard_of'] =  words['heard_of'].map(heardOfMap)
words['own_artist_music'] = words['own_artist_music'].map(ownMusicMap)
words.loc[words['heard_of']==0,'own_artist_music'] = 0

# duplicate 'good lyrics'
words['lyrics'] = words['good lyrics'].sum(axis=1)
words = words.drop(['good lyrics'], axis=1)
words = words.rename(index=str, columns={"lyrics":"good lyrics"})


words['positive'] = words[positiveWords].sum(axis=1)
words['negative'] = words[negativeWords].sum(axis=1)
words['neutral'] = words[neutralWords].sum(axis=1)

words['sentiment'] = words[['positive', 'negative', 'neutral']].idxmax(axis=1)
words = words.drop(positiveWords + negativeWords + neutralWords + ['positive', 'negative', 'neutral'], axis = 1)

"""
Preprocessing Users data file
"""
users = pd.read_csv('../Data/users.csv')
print "Users"
print len(users.columns)
users = users.rename(index=str, columns={'RESPID':'user'})

# Remove Q columns, Merge Q11 and Q12 to POP
cols = range(8,27)
users['POP'] = users[['Q11', 'Q12']].mean(axis = 1).round().map(rating)
users = users.drop(users.columns[cols], axis = 1)

# Map MUSIC
musicMap = {'Music has no particular interest for me':0,'Music is no longer as important as it used to be to me':1, 'I like music but it does not feature heavily in my life':2,'Music is important to me but not necessarily more important than other hobbies or interests':3, 'Music is important to me but not necessarily more important':3,'Music means a lot to me and is a passion of mine':4 }
users['MUSIC'] = users['MUSIC'].map(musicMap)

# Merge LIST_OWN and LIST_BACK to LIST
cols = ['LIST_OWN', 'LIST_BACK']
hoursMap = {'1 hour':1, 'More than 16 hours': 17, '16+ hours':17, 'Less than an hour':1}
for i in range(17):
	hoursMap[str(i)+' hours'] = i
for col in cols:
	users[col] = users[col].map(hoursMap)

users['LIST'] = users['LIST_OWN'] + users['LIST_BACK']
users = users.drop(cols, axis = 1)

# Handle LIST Missing Data based on MUSIC
for group in users.groupby('MUSIC'):
    nanLoc = group[1][group[1]['LIST'].isnull()].index
    meanVal = np.mean(group[1]['LIST'])
    users.loc[nanLoc,'LIST'] = np.random.poisson(meanVal, len(nanLoc))      #Poisson Distribution so that vals>=0
   
# AGE Missing Data
#users['AGE'].hist()
bin_range = np.arange(0,110,10)
out = pd.cut(users['AGE'], bins = bin_range, include_lowest = True, right = False, labels= np.arange(len(bin_range)-1))
dist = out.value_counts(sort = False).values
dist = dist/np.sum(dist)

nanLoc = users[users['AGE'].isnull()].index
ageBins = np.argmax(np.random.multinomial(1, dist, len(nanLoc)), axis=1)
out[nanLoc] = ageBins
users.loc[:, 'AGE'] = np.array(out)


# Group Region
users.loc[users['REGION']=='North Ireland','REGION']='Northern Ireland'
users.loc[users['REGION']=='Centre','REGION']='Midlands'

# REGION Missing Data
#users['REGION'].hist()
regions = users['REGION'].value_counts(sort = False)
dist = regions.values
dist = regions/np.sum(dist)

nanLoc = users[users['REGION'].isnull()].index
labels = np.argmax(np.random.multinomial(1, dist, len(nanLoc)), axis=1)
users.loc[nanLoc, 'REGION'] = regions.index[labels]

print users.loc[nanLoc,'REGION']

print users.head()
print users.isnull().sum()

data = pd.merge(users, words, on='user')

print trainData.columns


trainData = pd.merge(trainData, data, on='user')


testData = pd.merge(testData, data, on='user')


trainData.to_csv('../Data/wekaTrainingData.csv', index=False)
testData.to_csv('../Data/wekaTestData.csv', index=False)