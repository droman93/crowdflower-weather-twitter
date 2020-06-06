# import things
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

train_data = pd.read_csv('train.csv')
reviews_train = train_data['tweet']
train_sentiment = train_data[['s1','s2','s3','s4','s5']].values
train_sentiment[train_sentiment>0.5] = 1
train_sentiment[train_sentiment<0.5] = 0
train_sentiment = train_sentiment.astype('int64')


train_when = train_data[['w1','w2','w3','w4']].values
train_when[train_when>0.5] = 1
train_when[train_when<0.5] = 0
train_when = train_when.astype('int64')

train_kind = train_data[['k1','k2','k3','k4','k5',
                         'k6','k7','k8','k9','k10',
                         'k11','k12','k13','k14','k15']].values
train_kind[train_kind>0.5] = 1
train_kind[train_kind<0.5] = 0
train_kind = train_kind.astype('int64')



train_data.isnull().sum(axis=0) #checkin for nan values


test_data = pd.read_csv('test.csv')
ID = test_data['id'].values
reviews_test = test_data['tweet']
# on tip to use state wheter statics to help in predicitons, but this is a bit of overkill

# Trying first model non-linear classification
# First thought is to train 3 differnt models for each category  (s,w,k)
# model_s
# model_w
# model_k

# Import various modules for string cleaning
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
def review_to_wordlist( review, remove_stopwords=False ):
    # Function to convert a document to a sequence of words,
    # optionally removing stop words.  Returns a list of words.
    #
    # 1. Remove HTML
    review_text = BeautifulSoup(review).get_text()
    #
    # 2. Remove non-letters
    review_text = re.sub("[^a-zA-Z/d]"," ", review_text)
    #
    # 3. Convert words to lower case and split them
    words = review_text.lower().split()
    #
    # 4. Optionally remove stop words (false by default)
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
    #
    # 5. Return a list of words
    return(words)
def review_to_sentences( review, tokenizer, remove_stopwords=False ):
    # Function to split a review into parsed sentences. Returns a
    # list of sentences, where each sentence is a list of words
    #
    # 1. Use the NLTK tokenizer to split the paragraph into sentences
    raw_sentences = tokenizer.tokenize(review.strip())
    #
    # 2. Loop over each sentence
    sentences = []
    for raw_sentence in raw_sentences:
        # If a sentence is empty, skip it
        if len(raw_sentence) > 0:
            # Otherwise, call review_to_wordlist to get a list of words
            sentences.append( review_to_wordlist( raw_sentence, \
              remove_stopwords ))
    #
    # Return the list of sentences (each sentence is a list of words,
    # so this returns a list of lists
    return sentences
import nltk.data
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

tweets = []
for each_tweet in reviews_train:
    tweets += review_to_sentences(each_tweet,tokenizer)

len(tweets)
for each_tweet in reviews_test:
    tweets += review_to_sentences(each_tweet,tokenizer)


# Set values for various parameters
num_features = 300    # Word vector dimensionality
min_word_count = 40   # Minimum word count
num_workers = 6       # Number of threads to run in parallel
context = 10          # Context window size
downsampling = 1e-3   # Downsample setting for frequent words

# Initialize and train the model (this will take some time)
from gensim.models import word2vec
"Training model..."
model = word2vec.Word2Vec(tweets, workers=num_workers, \
            size=num_features, min_count = min_word_count, \
            window = context, sample = downsampling)

model.most_similar("bad")

import Gensim

def makeFeatureVec(words, model, num_features):
    # Function to average all of the word vectors in a given
    # paragraph
    #
    # Pre-initialize an empty numpy array (for speed)
    featureVec = np.zeros((num_features,),dtype="float32")
    #
    nwords = 0.
    #
    # Index2word is a list that contains the names of the words in
    # the model's vocabulary. Convert it to a set, for speed
    index2word_set = set(model.wv.index2word)
    #
    # Loop over each word in the review and, if it is in the model's
    # vocaublary, add its feature vector to the total
    for word in words:
        if word in index2word_set:
            nwords = nwords + 1.
            featureVec = np.add(featureVec,model[word])
    #
    # Divide the result by the number of words to get the average
    featureVec = np.divide(featureVec,nwords)
    return featureVec
def getAvgFeatureVecs(reviews, model, num_features):
    # Given a set of reviews (each one a list of words), calculate
    # the average feature vector for each one and return a 2D numpy array
    #
    # Initialize a counter
    counter = 0
    #
    # Preallocate a 2D numpy array, for speed
    reviewFeatureVecs = np.zeros((len(reviews),num_features),dtype="float32")
    #
    # Loop through the reviews
    for review in reviews:
       #
       # Print a status message every 1000th review
       if counter%1000. == 0.:
           print
           "Review %d of %d" % (counter, len(reviews))
       #
       # Call the function (defined above) that makes average feature vectors
       reviewFeatureVecs[counter] = makeFeatureVec(review, model,num_features)
       #
       # Increment the counter
       counter +=1
    return reviewFeatureVecs


train_tweets = []
for each_tweet in reviews_train:
    train_tweets.append(review_to_wordlist(each_tweet,remove_stopwords=True))

test_tweets = []
for each_tweet in reviews_test:
    test_tweets.append(review_to_wordlist(each_tweet,remove_stopwords=True))


trainData_vec = getAvgFeatureVecs(train_tweets,model,num_features)
trainData_vec = np.nan_to_num(trainData_vec)
# np.any(np.isnan(trainData_vec))
# np.any(np.isfinite(trainData_vec))

testData_vec = getAvgFeatureVecs(test_tweets,model,num_features)
testData_vec = np.nan_to_num(testData_vec)
# np.any(np.isnan(testData_vec))
# np.any(np.isfinite(testData_vec))


trainData_vec.fillna(trainData_vec.mean()) # for dataframe
np.isnan(trainData_vec.values.any()) # for dataframe


# Fit a random forest to the training data, using 100 trees
# using binary relevance
from skmultilearn.problem_transform import BinaryRelevance
from sklearn.datasets import make_multilabel_classification

from sklearn.ensemble import RandomForestClassifier
model_s = BinaryRelevance(RandomForestClassifier(n_estimators = 15,criterion='entropy'))
model_w = BinaryRelevance(RandomForestClassifier(n_estimators = 15,criterion='entropy'))
model_k = BinaryRelevance(RandomForestClassifier(n_estimators = 15,criterion='entropy'))

model_s = model_s.fit(trainData_vec,train_sentiment)
model_w = model_w.fit(trainData_vec,train_when)
model_k = model_k.fit(trainData_vec,train_kind)

# Test & extract results
S_res = model_s.predict_proba(testData_vec)
W_res = model_w.predict_proba(testData_vec)
K_res = model_k.predict_proba(testData_vec)

import scipy
S_res =  scipy.sparse.lil_matrix.toarray(S_res)
W_res =  scipy.sparse.lil_matrix.toarray(W_res)
K_res =  scipy.sparse.lil_matrix.toarray(K_res)


ID = S_res[:,1]
testData_vec.shape
# Output for submission
Sub = {'id':ID,'s1':S_res[:,0]
    ,'s2':S_res[:,1],'s3':S_res[:,2],'s4':S_res[:,3],'s5':S_res[:,4],
       'w1':W_res[:,0],'w2':W_res[:,1],'w3':W_res[:,2],'w4':W_res[:,3],
       'k1':K_res[:,0],'k2':K_res[:,1],'k3':K_res[:,2],'k4':K_res[:,3],'k5':K_res[:,4],'k6':K_res[:,5],
       'k7':K_res[:,6],'k8':K_res[:,7],'k9':K_res[:,8],'k10':K_res[:,9],'k11':K_res[:,10],
       'k12':K_res[:,11],'k13':K_res[:,12],'k14':K_res[:,13],'k15':K_res[:,14]}

Sub = pd.DataFrame(Sub)
Sub.to_csv('Submission_decizion_tree.csv', index=False)