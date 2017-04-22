import pandas as pd 
import sys
import numpy as np 

trainData = pd.read_csv('../Data/train.csv')
trainData.columns = [c.lower() for c in trainData.columns]

rating = {}
for i in range(25):
	rating[i]=0
for i in range(25,50):
	rating[i]=1
for i in range(50,75):
	rating[i]=2
for i in range(75,100):
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
print len(words.columns)
words = words.fillna(value=0)

print "Words"
# print (words.columns)
# print words.describe()

heardOfMap = {'Heard of':1, 'Heard of and listened to music EVER':1, 'Heard of and listened to music RECENTLY':1, 'Ever heard music by':1, 'Listened to recently':1  , 'Ever heard of ':1, 'Never heard of':0}
ownMusicMap = {'Own a lot of their music':2,'Own all or most of their music':2, 'Own a little of their music':1, 'Dont know':1, 'Own none of their music':0}

with open('../Data/positiveWords.txt') as f:
    content = f.readlines()
# you may also want to remove whitespace characters like `\n` at the end of each line
positiveWords = [x.strip().lower() for x in content] 

with open('../Data/negativeWords.txt') as f:
    content = f.readlines()
# you may also want to remove whitespace characters like `\n` at the end of each line
negativeWords = [x.strip().lower() for x in content] 

with open('../Data/neutralWords.txt') as f:
    content = f.readlines()
# you may also want to remove whitespace characters like `\n` at the end of each line
neutralWords = [x.strip().lower() for x in content] 

# Create new db, group heard of yes/no, own artist music, like rating, words list
# group by artist
words['heard_of'] =  words['heard_of'].map(heardOfMap)
words['own_artist_music'] = words['own_artist_music'].map(ownMusicMap)

# duplicate 'good lyrics'
words['lyrics'] = words['good lyrics'].sum(axis=1)
words = words.drop(['good lyrics'], axis=1)
words = words.rename(index=str, columns={"lyrics":"good lyrics"})

"""
words['sentiment'] = ''
for (index, row) in words.iterrows():
	for (wordList, label) in zip([positiveWords, negativeWords, neutralWords], ['Pos','Neg', 'Neut']):
		for w in wordList:
			if row[w]==1:
				words.ix[index,w] = ''.join([row['sentiment'],label])

"""
# cols = ['heard_of', 'own_artist_music']
# for col in cols:
#	print words[col].value_counts()



"""
Preprocessing Users data file
"""
users = pd.read_csv('../Data/users.csv')
print "Users"
print len(users.columns)
users = users.rename(index=str, columns={'RESPID':'user'})

cols = ['LIST_OWN', 'LIST_BACK']
hoursMap = {'1 hour':1, 'More than 16 hours': 17, '16+ hours':17, 'Less than an hour':0}
for i in range(17):
	hoursMap[str(i)+' hours'] = i
for col in cols:
	users[col] = users[col].map(hoursMap)

musicMap = {'Music has no particular interest for me':0,'Music is no longer as important as it used to be to me':1, 'I like music but it does not feature heavily in my life':2,'Music is important to me but not necessarily more important than other hobbies or interests':3, 'Music is important to me but not necessarily more important':3,'Music means a lot to me and is a passion of mine':4 }
users['MUSIC'] = users['MUSIC'].map(musicMap)

print users.head()

data = pd.merge(users, words, on='user')

print trainData.columns


trainData = pd.merge(trainData, data, on='user')


testData = pd.merge(testData, data, on='user')


trainData.to_csv('../Data/wekaTrainingData.csv', index=False)
testData.to_csv('../Data/wekaTestData.csv', index=False)