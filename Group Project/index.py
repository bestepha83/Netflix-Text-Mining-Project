import praw
reddit = praw.Reddit(client_id='08D24AM5aG_pj3_J4DFN1Q', # App ID
                     client_secret='lu3C66Ps1r1lNRt1D0Lf9RTgCsxz4w', # App password
                     user_agent='Text Miner 1.0', # App name (can be different)
                     username='Separate_Objective78', # Username
                     password='Awesome5') # User password


import sqlite3
import pandas as pd
import os
import re
import ast

df = pd.read_csv('comments_table.csv', encoding = 'latin-1')

conn = sqlite3.connect('comments.db')
# df.to_sql('df', conn) # CREATED THE SQL DATABASE


## CREATE THE TABLES THREADS AND COMMENTS ##

# conn = sqlite3.connect('Netflix.db')
# cur = conn.cursor()


# delete_sql1 = """DROP TABLE threads;"""
# delete_sql2 = """DROP TABLE comments;"""
# create_sql1 = """CREATE TABLE threads (ID text primary key, title text, body text, author text, url text, 
#                 created_utc real, num_comments integer, score integer);"""
# create_sql2 = """CREATE TABLE comments (ID text primary key, parent_ID text, thread_ID text,
#                 body text, author text, created_utc real, score integer,
#                 foreign key (ID) references threads(ID));"""

# cur.execute(delete_sql1)
# cur.execute(delete_sql2)
# cur.execute(create_sql1)
# cur.execute(create_sql2)


# p = re.compile(r'netflix|nflx', re.IGNORECASE)

# ## INSERT DATA RECORDS ##
# subreddit1 = reddit.subreddit('technology')
# # for thread in subreddit.search(query=p.findall(thread)):
# for thread in subreddit1.search(query='netflix'):
#     sql = "INSERT INTO threads VALUES (?, ?, ?, ?, ?, ?, ?, ?);"
#     cur.execute(sql, (thread.id, thread.title, thread.selftext, thread.author.name, 
#                       thread.url, thread.created_utc, thread.num_comments, thread.score))
#     thread.comments.replace_more(limit = 0)
#     comment_num = 0
#     for comment in thread.comments:
#         comment_num += 1
#         if comment_num > 5:
#             break
#         if comment.author != None:
#             sql = "INSERT INTO comments VALUES (?, ?, ?, ?, ?, ?, ?);"
#             cur.execute(sql, (comment.id, comment.parent_id, comment.link_id, comment.body, 
#                             comment.author.name, comment.created_utc, comment.score))

# subreddit2 = reddit.subreddit('stocks')
# for thread in subreddit2.search(query='netflix'):
#     sql = "INSERT INTO threads VALUES (?, ?, ?, ?, ?, ?, ?, ?);"
#     cur.execute(sql, (thread.id, thread.title, thread.selftext, thread.author.name, 
#                       thread.url, thread.created_utc, thread.num_comments, thread.score))
#     thread.comments.replace_more(limit = 0)
#     comment_num = 0
#     for comment in thread.comments:
#         comment_num += 1
#         if comment_num > 5:
#             break
#         if comment.author != None:
#             sql = "INSERT INTO comments VALUES (?, ?, ?, ?, ?, ?, ?);"
#             cur.execute(sql, (comment.id, comment.parent_id, comment.link_id, comment.body, 
#                             comment.author.name, comment.created_utc, comment.score))

# print("Done creating database!")

## CREATE CSV ##
# query_threads_sql = "SELECT * FROM threads;"
# threads_db = pd.read_sql(query_threads_sql, conn)
# threads_db.to_csv('threads_table.csv', encoding='utf-8')

# query_comments_sql = "SELECT * FROM df;"
# comments_db = pd.read_sql(query_comments_sql, conn)
# print(comments_db)


## ADD RESULTS TO LISTS ##
cur = conn.cursor()

review_list = []
score_lst = []

cur.execute("SELECT body FROM df;")
for row in cur.fetchall():
    review_list.append(''.join(row))

cur.execute("SELECT score FROM df;")
for row in cur.fetchall():
    row = int(row[0])
    score_lst.append(row)



# Expanding contraction
# ref: https://mlwhiz.com/blog/2019/01/17/deeplearning_nlp_preprocess/
contraction_dict = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have",
                    "couldn't": "could not", "didn't": "did not",  "doesn't": "does not", "don't": "do not", 
                    "hadn't": "had not", "hasn't": "has not", "haven't": "have not", "he'd": "he would","he'll": "he will", 
                    "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",  
                    "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", 
                    "I've": "I have", "i'd": "i would", "i'd've": "i would have", "i'll": "i will",  
                    "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would", 
                    "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", 
                    "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have",
                    "mightn't": "might not","mightn't've": "might not have", "must've": "must have", "mustn't": "must not", 
                    "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have",
                    "o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", 
                    "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have", 
                    "she'd": "she would", "she'd've": "she would have", "she'll": "she will", 
                    "she'll've": "she will have", "she's": "she is", "should've": "should have", 
                    "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have",
                    "so's": "so as", "this's": "this is","that'd": "that would", "that'd've": "that would have", 
                    "that's": "that is", "there'd": "there would", "there'd've": "there would have", "there's": "there is", 
                    "here's": "here is","they'd": "they would", "they'd've": "they would have", "they'll": "they will", 
                    "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have", 
                    "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", 
                    "we'll've": "we will have", "we're": "we are", "we've": "we have", "weren't": "were not", 
                    "what'll": "what will", "what'll've": "what will have", "what're": "what are",  "what's": "what is", 
                    "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", 
                    "where's": "where is", "where've": "where have", "who'll": "who will", "who'll've": "who will have", 
                    "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have", 
                    "will've": "will have", "won't": "will not", "won't've": "will not have", "would've": "would have", 
                    "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all", 
                    "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are",
                    "y'all've": "you all have","you'd": "you would", "you'd've": "you would have", "you'll": "you will", 
                    "you'll've": "you will have", "you're": "you are", "you've": "you have"}

def _get_contractions(contraction_dict):
    contraction_re = re.compile('(%s)' % '|'.join(contraction_dict.keys()))
    return contraction_dict, contraction_re

contractions, contractions_re = _get_contractions(contraction_dict)

def replace_contractions(text):
    def replace(match):
        return contractions[match.group(0)]
    return contractions_re.sub(replace, text)

# tokenization
from nltk.tokenize import RegexpTokenizer

def tokenize(text):
    tokenizer = RegexpTokenizer(r'[\w\']+|\$[\d\.]+|\S+')
    return tokenizer.tokenize(text)


# stop words
def rv_stopwords(tokenized_text):
    sw_list = [",", ".", "'", '"', 'a', 'the', 'from', 'and', 'in', 'to', 'or', \
        'are', 'that', 'this', 'than', 'now', 'after', 'which', 'will', 'they', \
            'their', 'is']
    return [word for word in tokenized_text if word not in sw_list]

def preprocess(text):
    text = replace_contractions(text)
    text = text.lower()
    tokens = tokenize(text)
    tokens = rv_stopwords(tokens)
    return tokens


corpus = []

for idx, text in enumerate(review_list):
    corpus.append(preprocess(text))
print("Finished preprocessing!")

import gensim.models

print("Model training...")
model = gensim.models.Word2Vec(sentences=corpus, sg=0, vector_size=100, window=5)
print("Finished model training!")




## UNSUPERVISED LEARNING ##

# Get the data

from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()  # for plot styling
import numpy as np

X, y_true = make_blobs(n_samples=300, centers=4,
                       cluster_std=0.60, random_state=0)
plt.scatter(X[:, 0], X[:, 1], s=50);

# K-means; k = 10
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=10)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')

centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)


word_list = []
wv_list = []

for t in model.wv.most_similar(positive=['good'], topn = 50):
    word_list.append(t[0])
    wv_list.append(model.wv[t[0]])
for t in model.wv.most_similar(positive=['up'], topn = 50):
    word_list.append(t[0])
    wv_list.append(model.wv[t[0]])
for t in model.wv.most_similar(positive=['growth'], topn = 50):
    word_list.append(t[0])
    wv_list.append(model.wv[t[0]])


wv_list = np.array(wv_list)

kmeans = KMeans(n_clusters=3)
kmeans.fit(wv_list)
y_kmeans = kmeans.predict(wv_list)
print(y_kmeans)


# viz

from sklearn.manifold import TSNE                   
import random

num_dimensions = 2

# reduce using t-SNE
tsne = TSNE(n_components=num_dimensions, random_state=0)
vectors = tsne.fit_transform(wv_list)

x_vals = [v[0] for v in vectors]
y_vals = [v[1] for v in vectors]

indices = list(range(len(word_list)))
selected_indices = random.sample(indices, 30)

plt.figure(figsize=(12, 12))
for i in selected_indices:
    plt.annotate(word_list[i], (x_vals[i], y_vals[i]))
    
plt.scatter(x_vals, y_vals, c=y_kmeans, s=30, cmap='viridis')


# Association rule mining
# We want to mine association between different movie actors/actress
# We just want names from the text <- part-of-speech tagging: proper nouns: NNP

import nltk
# nltk.download('averaged_perceptron_tagger')
# print(review_list[10])

tokenizer = RegexpTokenizer(r'[\w\']+|\$[\d\.]+|\S+')
# print(review_list[10])
# tokenized_text = tokenizer.tokenize(review_list["reviewText"])
# print(nltk.pos_tag(tokenized_text))


def arm_preprocess(text):
    text = replace_contractions(text)
    text = text.lower()
    tokens = RegexpTokenizer(r'[a-zA-Z]+').tokenize(text)
    tokens = [t[0] for t in nltk.pos_tag(tokens) if t[1] == "NNP"]
    return tokens


arm_corpus = []

for idx, text in enumerate(review_list):
    if(idx % 1000 == 0):
        print("{0} files have been preprocessed.".format(idx))
    arm_corpus.append(arm_preprocess(text))

print("Finished preprocessing!")


print(arm_corpus)


from apyori import apriori

associations = apriori(arm_corpus, min_length = 2, max_length = 4, min_support = 0.0001, min_confidence = 0.1, min_lift = 2)
print(list(associations))

plt.show()








# ## SUPERVISED LEARNING ##

# # Prepare input and output for linear regression
# X = []
# Y = []

# for t in model.wv.most_similar(positive=['good'], topn = 300):
#     # t is a tuple, t[0] is the word, t[1] is corresponding similarity
#     # We want word vector as input variables
#     # Word vectors can be obtained by model.wv[word]
    
#     X.append(model.wv[t[0]])
#     Y.append(t[1])

# # split dataset
# from sklearn.model_selection import train_test_split
# import numpy as np


# # convert X to numpy arrays
# X = np.array(X)
# Y = np.array(Y)
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25)

# print(X)

# print(X_train.shape)
# print(X_test.shape)

# from sklearn.linear_model import LinearRegression

# #fit the model
# reg = LinearRegression().fit(X_train, Y_train)
# print(reg.coef_)

# # Use the trained model to predict new values
# Y_pred = reg.predict(X_test)


# print(Y_pred)

# # Evaluate your model
# from sklearn import metrics

# print(metrics.mean_squared_error(Y_test, Y_pred))

# # What about another model:
# # Just the first 5 dimensions of the word vectors
# X = []
# Y = []

# for t in model.wv.most_similar(positive=['good'], topn = 100):
#     X.append(model.wv[t[0]][:5])
#     Y.append(t[1])
# X = np.array(X)
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25)
# reg = LinearRegression().fit(X_train, Y_train)
# Y_pred = reg.predict(X_test)
# print(metrics.mean_squared_error(Y_test, Y_pred))

# # k-fold cross-validation
# from sklearn.model_selection import KFold

# kf = KFold(n_splits=10)
# for train_index, test_index in kf.split(X):
#     print("TRAIN:", train_index, "TEST:", test_index)
#     print("\n")
#     print(type(train_index), type(test_index))
#     print(train_index)
#     X_train, X_test = X[train_index], X[test_index]
#     Y_train, Y_test = Y[train_index], Y[test_index]

# # You can shuffle the dataset to make it even more randomized
# # random_state in some other functions also called random_seed
# # will generate the same randomized results -> When you want your results to be repeatable

# kf = KFold(n_splits=10, shuffle=True, random_state=11)
# for train_index, test_index in kf.split(X):
#     print("TRAIN:", train_index, "TEST:", test_index)
#     print("\n")
#     X_train, X_test = X[train_index], X[test_index]
#     Y_train, Y_test = Y[train_index], Y[test_index]



# # Now you can train your model, calculate MSE, then average the score to evaluate the model

# X = []
# Y = []

# for t in model.wv.most_similar(positive=['good'], topn = 100):
#     X.append(model.wv[t[0]][:5])
#     Y.append(t[1])
# X = np.array(X)
# Y = np.array(Y)


# MSE = []

# kf = KFold(n_splits=10, shuffle=True, random_state=11)
# for train_index, test_index in kf.split(X):
#     X_train, X_test = X[train_index], X[test_index]
#     Y_train, Y_test = Y[train_index], Y[test_index]
#     reg = LinearRegression().fit(X_train, Y_train)
#     Y_pred = reg.predict(X_test)
#     MSE.append(metrics.mean_squared_error(Y_test, Y_pred))
# print(np.mean(MSE))



# X = []
# Y = []

# for t in model.wv.most_similar(positive=['good'], topn = 100):
#     X.append(model.wv[t[0]])
#     Y.append(t[1])
# X = np.array(X)
# Y = np.array(Y)


# MSE = []

# kf = KFold(n_splits=10, shuffle=True, random_state=11)
# for train_index, test_index in kf.split(X):
#     X_train, X_test = X[train_index], X[test_index]
#     Y_train, Y_test = Y[train_index], Y[test_index]
#     reg = LinearRegression().fit(X_train, Y_train)
#     Y_pred = reg.predict(X_test)
#     MSE.append(metrics.mean_squared_error(Y_test, Y_pred))
# print(np.mean(MSE))


# # There might be overfitting problems
# # So maybe just the first 5 features and you can get a much better model


# # You can even compare your choice of how many features you want
# # and then plot the results in a line chart

# import matplotlib.pyplot as plt

# line_chart_x = []
# line_chart_y = []


# for i in range(5, 100):
#     X = []
#     Y = []

#     for t in model.wv.most_similar(positive=['good'], topn = 100):
#         X.append(model.wv[t[0]][:i])
#         Y.append(t[1])
#     X = np.array(X)
#     Y = np.array(Y)


#     MSE = []

#     kf = KFold(n_splits=10, shuffle=True, random_state=11)
#     for train_index, test_index in kf.split(X):
#         X_train, X_test = X[train_index], X[test_index]
#         Y_train, Y_test = Y[train_index], Y[test_index]
#         reg = LinearRegression().fit(X_train, Y_train)
#         Y_pred = reg.predict(X_test)
#         MSE.append(metrics.mean_squared_error(Y_test, Y_pred))
    
#     line_chart_x.append(i)
#     line_chart_y.append(np.mean(MSE))

# plt.plot(line_chart_x, line_chart_y)
# plt.show()


# # binary classification: logistic regression
# # comparison between the Godfather and the Matrix 


# X = []
# Y = []

# for t in model.wv.most_similar(positive=['growth'], topn = 300):
#     X.append(model.wv[t[0]])
#     Y.append(1)
# for t in model.wv.most_similar(positive=['drop'], topn = 300):
#     X.append(model.wv[t[0]])
#     Y.append(0)

# X = np.array(X)
# Y = np.array(Y)

# # This might be problematic, a word could be in the first list as well as the second list

# # build a dictionary to store the data first

# word_vector = {}

# for t in model.wv.most_similar(positive=['growth'], topn = 300):
#     word_vector[t[0]] = [model.wv[t[0]], 1]
# for t in model.wv.most_similar(positive=['drop'], topn = 300):
#     # if the word already exists in the dictionary
#     if(t[0] in word_vector):
#         # label it with another value
#         word_vector[t[0]][1] = -1
#     else:
#         word_vector[t[0]] = [model.wv[t[0]], 0]

# print(word_vector)


# # Now add it to X and Y

# X = []
# Y = []


# double_exist_count = 0

# for key, value in word_vector.items():
#     if(value[1] == 1):
#         X.append(value[0])
#         Y.append(1)
#     elif(value[1] == 0):
#         X.append(value[0])
#         Y.append(0)
#     else:
#         double_exist_count += 1
#         continue

# print(double_exist_count)


# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25)

# # fit the model
# from sklearn.linear_model import LogisticRegression


# clf = LogisticRegression(random_state=0).fit(X_train, Y_train)

# print(clf.predict(X_test))


# from sklearn.metrics import confusion_matrix

# # Get the value of true positive, false positive, false negative, and true negative
# print(confusion_matrix(Y_test, clf.predict(X_test)))
# tn, fp, fn, tp = confusion_matrix(Y_test, clf.predict(X_test)).ravel()
# print(tn, fp, fn, tp)


# # calculate precision, recall, and f1
# precision = tp/(tp+fp)
# recall = tp/(tp+fn)
# f1 = 2*precision*recall/(precision+recall)
# print("precision: ", precision)
# print("recall:", recall)
# print("f1: ", f1)


# # Now use the trained model to predict new values

# movies = ["batman", "transformers", "gump", "goodfellas", "casablanca", "shawshank"]

# # label 1 -- the Godfather 
# # label 0 -- the Matrix

# comparison = ["matrix", "godfather"]

# for m in movies:
#     prediction_result = clf.predict([model.wv[m]])[0]
#     # print(prediction_result)
#     print("{0:20}{1:20}{2:20}".format(m, "--------",comparison[prediction_result]))


# # 10-fold cross-validation

# word_vector = {}

# for t in model.wv.most_similar(positive=['godfather'], topn = 300):
#     word_vector[t[0]] = [model.wv[t[0]], 1]
# for t in model.wv.most_similar(positive=['matrix'], topn = 300):
#     # if the word already exists in the dictionary
#     if(t[0] in word_vector):
#         # label it with another value
#         word_vector[t[0]][1] = -1
#     else:
#         word_vector[t[0]] = [model.wv[t[0]], 0]

# X = []
# Y = []

# for key, value in word_vector.items():
#     if(value[1] == 1):
#         X.append(value[0])
#         Y.append(1)
#     elif(value[1] == 0):
#         X.append(value[0])
#         Y.append(0)
#     else:
#         double_exist_count += 1
#         continue

# X = np.array(X)
# Y = np.array(Y)

# f1_list = []

# kf = KFold(n_splits=10, shuffle=True, random_state=11)
# for train_index, test_index in kf.split(X):
#     X_train, X_test = X[train_index], X[test_index]
#     Y_train, Y_test = Y[train_index], Y[test_index]
#     clf = LogisticRegression(random_state=0).fit(X_train, Y_train)
#     tn, fp, fn, tp = confusion_matrix(Y_test, clf.predict(X_test)).ravel()
    
#     precision = tp/(tp+fp)
#     recall = tp/(tp+fn)
#     f1 = 2*precision*recall/(precision+recall)
    
#     f1_list.append(f1)
# print(np.mean(f1_list))

# # Support vector machine

# from sklearn import svm

# # Just change the function name
# clf = svm.SVC().fit(X_train, Y_train)
# tn, fp, fn, tp = confusion_matrix(Y_test, clf.predict(X_test)).ravel()

# precision = tp/(tp+fp)
# recall = tp/(tp+fn)
# f1 = 2*precision*recall/(precision+recall)
# print("precision: ", precision)
# print("recall:", recall)
# print("f1: ", f1)


# # kernerl tricks
# for kernel in ('linear', 'poly', 'rbf'):
#     print("current kernel: " + kernel)
#     clf = svm.SVC(kernel=kernel, gamma=2).fit(X_train, Y_train)
#     tn, fp, fn, tp = confusion_matrix(Y_test, clf.predict(X_test)).ravel()
#     precision = tp/(tp+fp)
#     recall = tp/(tp+fn)
#     f1 = 2*precision*recall/(precision+recall)
#     print("precision: ", precision)
#     print("recall:", recall)
#     print("f1: ", f1)
#     print("\n")
