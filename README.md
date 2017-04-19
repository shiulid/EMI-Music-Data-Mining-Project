# EMI-Music-Data-Mining-Project

Goal: Design an algorithm that combines user’s demographics, artist and track ratings, answers to questions about their preferences for music, and words that they use to describe EMI artists in order to predict how much they like tracks they have just heard.

Dataset: EMI One Million Interview Dataset is a large and rich music dataset that contains interests, attitudes, behaviors, familiarity and appreciation of music as expressed by music fans around the world.

Details:

Train/Test set – Artist ID, Track ID, User ID, Rating (X-100) (train to predict this for test set), Time (when market research was completed).

Words – Contains data that shows how people describe the artists they have heard, including a list of words such as ‘Soulful’ / ‘Aggressive’ etc describing the artist.

Users – User details such as demographics, age, etc and answers to some music habit questions.

In this project we will be treating the rating prediction as a classification task by binning the rating values. We will analyze the data, develop and evaluate an efficient way of predicting the rating feature bin. 
