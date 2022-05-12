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

# comments
df1 = pd.read_csv('comments_table.csv', encoding = 'latin-1')

comments_conn = sqlite3.connect('comments.db')
# df1.to_sql('comments_df', comments_conn) # CREATED THE SQL DATABASE

# threads
df2 = pd.read_csv('threads_table.csv', encoding = 'latin-1')

threads_conn = sqlite3.connect('threads.db')
# df2.to_sql('threads_df', threads_conn) # CREATED THE SQL DATABASE


# ## CREATE THE TABLES 'threads' AND 'comments' ##

# conn = sqlite3.connect('Netflix.db')
# cur = conn.cursor()


# delete_sql1 = """DROP TABLE threads;"""
# delete_sql2 = """DROP TABLE comments;"""
# create_sql1 = """CREATE TABLE threads (ID text primary key, title text, author text, url text, 
#                 created_utc real, num_comments integer, score integer);"""
# create_sql2 = """CREATE TABLE comments (ID text primary key, thread_ID text,
#                 body text, author text, created_utc real, score integer,
#                 foreign key (ID) references threads(ID));"""

# cur.execute(delete_sql1)
# cur.execute(delete_sql2)
# cur.execute(create_sql1)
# cur.execute(create_sql2)


# ## INSERT DATA RECORDS ##

# # Technology Subreddit
# thread_num = 0
# subreddit1 = reddit.subreddit('technology')
# for thread in subreddit1.search(query='netflix'):
#     sql = "INSERT INTO threads VALUES (?, ?, ?, ?, ?, ?, ?);"
#     cur.execute(sql, (thread.id, thread.title, thread.author.name, 
#                       thread.url, thread.created_utc, thread.num_comments, thread.score))
#     thread.comments.replace_more(limit = 0)

#     thread_num += 1
#     if(thread_num % 10 == 0):
#         print("{0} netflix thread files have been processed.".format(thread_num))

#     comment_num = 0
#     for comment in thread.comments:
#         comment_num += 1
#         if comment_num > 5:
#             break
#         if comment.author != None:
#             sql = "INSERT INTO comments VALUES (?, ?, ?, ?, ?, ?);"
#             cur.execute(sql, (comment.id, comment.link_id, comment.body, 
#                             comment.author.name, comment.created_utc, comment.score))

# thread_ids_lst = []
# thread_ids = cur.execute("SELECT ID FROM threads")
# for id in thread_ids.fetchall():
#     thread_ids_lst.append(''.join(id))

# thread_num = 0
# for thread in subreddit1.search(query='nflx'):
#     # NOTE: there are only 3 matches!
#     if thread.id not in thread_ids_lst:
#         sql = "INSERT INTO threads VALUES (?, ?, ?, ?, ?, ?, ?);"
#         cur.execute(sql, (thread.id, thread.title, thread.author.name, 
#                         thread.url, thread.created_utc, thread.num_comments, thread.score))
#         thread.comments.replace_more(limit = 0)

#         thread_num += 1
#         if(thread_num % 10 == 0):
#             print("{0} nflx thread files have been processed.".format(thread_num))

#         comment_num = 0
#         for comment in thread.comments:
#             comment_num += 1
#             if comment_num > 5:
#                 break
#             if comment.author != None:
#                 sql = "INSERT INTO comments VALUES (?, ?, ?, ?, ?, ?);"
#                 cur.execute(sql, (comment.id, comment.link_id, comment.body, 
#                                 comment.author.name, comment.created_utc, comment.score))

# # Stocks Subreddit
# thread_num = 0
# subreddit2 = reddit.subreddit('stocks')
# for thread in subreddit2.search(query='netflix'):
#     sql = "INSERT INTO threads VALUES (?, ?, ?, ?, ?, ?, ?);"
#     cur.execute(sql, (thread.id, thread.title, thread.author.name, 
#                       thread.url, thread.created_utc, thread.num_comments, thread.score))
#     thread.comments.replace_more(limit = 0)

#     thread_num += 1
#     if(thread_num % 10 == 0):
#         print("{0} netflix thread files have been processed.".format(thread_num))

#     comment_num = 0
#     for comment in thread.comments:
#         comment_num += 1
#         if comment_num > 5:
#             break
#         if comment.author != None:
#             sql = "INSERT INTO comments VALUES (?, ?, ?, ?, ?, ?);"
#             cur.execute(sql, (comment.id, comment.link_id, comment.body, 
#                             comment.author.name, comment.created_utc, comment.score))

# thread_ids_lst = []
# thread_ids = cur.execute("SELECT ID FROM threads")
# for id in thread_ids.fetchall():
#     thread_ids_lst.append(''.join(id))

# thread_num = 0
# for thread in subreddit2.search(query='nflx'):
#     if thread.id not in thread_ids_lst:
#         sql = "INSERT INTO threads VALUES (?, ?, ?, ?, ?, ?, ?);"
#         cur.execute(sql, (thread.id, thread.title, thread.author.name, 
#                         thread.url, thread.created_utc, thread.num_comments, thread.score))
#         thread.comments.replace_more(limit = 0)

#         thread_num += 1
#         if(thread_num % 10 == 0):
#             print("{0} nflx thread files have been processed.".format(thread_num))

#         comment_num = 0
#         for comment in thread.comments:
#             comment_num += 1
#             if comment_num > 5:
#                 break
#             if comment.author != None:
#                 sql = "INSERT INTO comments VALUES (?, ?, ?, ?, ?, ?);"
#                 cur.execute(sql, (comment.id, comment.link_id, comment.body, 
#                                 comment.author.name, comment.created_utc, comment.score))

# print("Done creating database!")

# # CREATE CSV ##
# query_threads_sql = "SELECT * FROM threads;"
# threads_db = pd.read_sql(query_threads_sql, conn)
# threads_db.to_csv('threads_table.csv', encoding='utf-8')

# query_comments_sql = "SELECT * FROM comments;"
# comments_db = pd.read_sql(query_comments_sql, conn)
# comments_db.to_csv('comments_table.csv', encoding='utf-8')


## ADD RESULTS TO LISTS ##

review_list = []
score_lst = []


threads_cur = threads_conn.cursor()

threads_cur.execute("SELECT title FROM threads_df;")
for row in threads_cur.fetchall():
    review_list.append(''.join(row))

threads_cur.execute("SELECT score FROM threads_df;")
for row in threads_cur.fetchall():
    row = int(row[0])
    score_lst.append(row)


comments_cur = comments_conn.cursor()

comments_cur.execute("SELECT body FROM comments_df;")
for row in comments_cur.fetchall():
    review_list.append(''.join(row))

comments_cur.execute("SELECT score FROM comments_df;")
for row in comments_cur.fetchall():
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
    #NOTE: we can add more preprocessing here towards a specific topic (advertisements, etc.)

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



# Prepare input and output for linear regression
X = []
Y = []

for t in model.wv.most_similar(positive=['grow'], topn = 300):
    print(t)
    # t is a tuple, t[0] is the word, t[1] is corresponding similarity
    # We want word vector as input variables
    # Word vectors can be obtained by model.wv[word]
    
    
    # t --> ('amusing', 0.79138192)
    # t[0] -> 'amusing'
    # t[1] -> cosine similarity
    
    # X <- word vectors
    # model.wv[t[0]]
    
    
    
    X.append(model.wv[t[0]])
    Y.append(t[1])

import numpy as np


# convert X and Y to numpy arrays

X = np.array(X)
Y = np.array(Y)


print(Y.shape)


# split dataset
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25)

print(Y.shape)
print(Y_train.shape)
print(Y_test.shape)

print(X_train.shape)
print(X_test.shape)


from sklearn.linear_model import LinearRegression


#fit the model
reg = LinearRegression().fit(X_train, Y_train)
print(reg.coef_)


# Use the trained model to predict new values
Y_pred = reg.predict(X_test)


print(Y_pred)

# Evaluate your model
from sklearn import metrics

print(metrics.mean_squared_error(Y_test, Y_pred))


# What about another model:
# Just the first 5 dimensions of the word vectors
X = []
Y = []

for t in model.wv.most_similar(positive=['grow'], topn = 300):
    X.append(model.wv[t[0]][:5])
    Y.append(t[1])
X = np.array(X)
Y = np.array(Y)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25)
reg = LinearRegression().fit(X_train, Y_train)
Y_pred = reg.predict(X_test)
print(metrics.mean_squared_error(Y_test, Y_pred))


# In[59]:


# k-fold cross-validation
from sklearn.model_selection import KFold


# In[60]:


kf = KFold(n_splits=10)
for train_index, test_index in kf.split(X):
    print("TRAIN:", train_index, "TEST:", test_index)
    print("\n")
    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]


# In[61]:


# You can shuffle the dataset to make it even more randomized
# random_state in some other functions also called random_seed
# will generate the same randomized results -> When you want your results to be repeatable
#
kf = KFold(n_splits=10, shuffle=True, random_state=11)
for train_index, test_index in kf.split(X):
    print("TRAIN:", train_index, "TEST:", test_index)
    print("\n")
    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]


# In[62]:


# Now you can train your model, calculate MSE, then average the score to evaluate the model

X = []
Y = []

for t in model.wv.most_similar(positive=['grow'], topn = 300):
    X.append(model.wv[t[0]][:5])
    Y.append(t[1])
X = np.array(X)
Y = np.array(Y)


MSE = []

kf = KFold(n_splits=10, shuffle=True, random_state=11)
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]
    reg = LinearRegression().fit(X_train, Y_train)
    Y_pred = reg.predict(X_test)
    MSE.append(metrics.mean_squared_error(Y_test, Y_pred))
print(np.mean(MSE))


# In[63]:


X = []
Y = []

for t in model.wv.most_similar(positive=['grow'], topn = 300):
    X.append(model.wv[t[0]])
    Y.append(t[1])
X = np.array(X)
Y = np.array(Y)


MSE = []

kf = KFold(n_splits=10, shuffle=True, random_state=11)
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]
    reg = LinearRegression().fit(X_train, Y_train)
    Y_pred = reg.predict(X_test)
    MSE.append(metrics.mean_squared_error(Y_test, Y_pred))
print(np.mean(MSE))


# In[32]:


# You can even compare your choice of how many features you want
# and then plot the results in a line chart

import matplotlib.pyplot as plt

line_chart_x = []
line_chart_y = []


for i in range(5, 300):
    X = []
    Y = []

    for t in model.wv.most_similar(positive=['grow'], topn = 300):
        X.append(model.wv[t[0]][:i])
        Y.append(t[1])
    X = np.array(X)
    Y = np.array(Y)


    MSE = []

    kf = KFold(n_splits=10, shuffle=True, random_state=11)
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]
        reg = LinearRegression().fit(X_train, Y_train)
        Y_pred = reg.predict(X_test)
        MSE.append(metrics.mean_squared_error(Y_test, Y_pred))
    
    line_chart_x.append(i)
    line_chart_y.append(np.mean(MSE))

plt.plot(line_chart_x, line_chart_y)
plt.show()







# # binary classification: logistic regression
# # grow vs. stratocaster 

# X = []
# Y = []

# for t in model.wv.most_similar(positive=['grow'], topn = 300):
#     X.append(model.wv[t[0]])
#     Y.append(1)
# for t in model.wv.most_similar(positive=['stratocaster'], topn = 300):
#     X.append(model.wv[t[0]])
#     Y.append(0)

# X = np.array(X)
# Y = np.array(Y)

# # This might be problematic, a word could be in the first list as well as the second list


# # In[65]:


# # build a dictionary to store the data first

# grow_dict = {}
# stratocaster_dict = {}

# for t in model.wv.most_similar(positive=['grow'], topn = 300):
#     grow_dict[t[0]] = model.wv[t[0]]
# for t in model.wv.most_similar(positive=['stratocaster'], topn = 300):
#     stratocaster_dict[t[0]] = model.wv[t[0]]


# # In[82]:


# # Now add it to X and Y

# X = []
# Y = []

# for word in grow_dict.keys():
#     if(word not in stratocaster_dict.keys()):
#         X.append(grow_dict[word])
#         Y.append(0)
        
# for word in stratocaster_dict.keys():
#     if(word not in grow_dict.keys()):
#         X.append(stratocaster_dict[word])
#         Y.append(1)


# # In[83]:


# print(len(X))
# print(Y)


# # In[84]:


# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25)


# # In[73]:


# # fit the model
# from sklearn.linear_model import LogisticRegression


# # In[85]:


# clf = LogisticRegression(random_state=0).fit(X_train, Y_train)

# print(clf.predict(X_test))


# # In[75]:


# from sklearn.metrics import confusion_stratocaster


# # In[86]:


# # Get the value of true positive, false positive, false negative, and true negative
# print(confusion_stratocaster(Y_test, clf.predict(X_test)))
# tn, fp, fn, tp = confusion_stratocaster(Y_test, clf.predict(X_test)).ravel()
# print(tn, fp, fn, tp)


# # In[87]:


# # calculate precision, recall, and f1
# precision = tp/(tp+fp)
# recall = tp/(tp+fn)
# f1 = 2*precision*recall/(precision+recall)
# print("precision: ", precision)
# print("recall:", recall)
# print("f1: ", f1)


# # In[89]:


# # Now use the trained model to predict new values

# words = ["stratocaster", "telecaster", "drums", "gibson", "fender", "ibanez",
#         "rich", "natural", "colorful", "lightweight", "unique"]

# # label 0 -- grow 
# # label 1 -- the stratocaster

# comparison = ["grow", "stratocaster"]

# for m in words:
#     prediction_result = clf.predict([model.wv[m]])[0]
#     # print(prediction_result)
#     print("{0:20}{1:20}{2:20}".format(m, "--------",comparison[prediction_result]))


# # In[95]:


# from sklearn.metrics import classification_report


# # In[101]:



# # 10-fold cross-validation


# grow_dict = {}
# stratocaster_dict = {}

# for t in model.wv.most_similar(positive=["grow"], topn = 300):
#     grow_dict[t[0]] = model.wv[t[0]]
# for t in model.wv.most_similar(positive=['stratocaster'], topn = 300):
#     stratocaster_dict[t[0]] = model.wv[t[0]]

# X = []
# Y = []

# for word in grow_dict.keys():
#     if(word not in stratocaster_dict.keys()):
#         X.append(grow_dict[word])
#         Y.append(0)
        
# for word in stratocaster_dict.keys():
#     if(word not in grow_dict.keys()):
#         X.append(stratocaster_dict[word])
#         Y.append(1)

# X = np.array(X)
# Y = np.array(Y)
        
# f1_list = []

# kf = KFold(n_splits=10, shuffle=True, random_state=11)
# for train_index, test_index in kf.split(X):
#     X_train, X_test = X[train_index], X[test_index]
#     Y_train, Y_test = Y[train_index], Y[test_index]
#     clf = LogisticRegression(random_state=0).fit(X_train, Y_train)
#     tn, fp, fn, tp = confusion_stratocaster(Y_test, clf.predict(X_test)).ravel()
    
#     precision = tp/(tp+fp)
#     recall = tp/(tp+fn)
#     f1 = 2*precision*recall/(precision+recall)
    
#     print(classification_report(Y_test, clf.predict(X_test)))
    
#     f1_list.append(f1)
# print(np.mean(f1_list))


# # In[140]:


# from sklearn.metrics import precision_recall_fscore_support
# from sklearn.metrics import accuracy_score


# # In[143]:


# # 3 class classification


# grow_dict = {}
# stratocaster_dict = {}
# freeman_dict = {}

# for t in model.wv.most_similar(positive=['grow'], topn = 300):
#     grow_dict[t[0]] = model.wv[t[0]]
# for t in model.wv.most_similar(positive=['stratocaster'], topn = 300):
#     stratocaster_dict[t[0]] = model.wv[t[0]]
# for t in model.wv.most_similar(positive=['freeman'], topn = 300):
#     freeman_dict[t[0]] = model.wv[t[0]]

# X = []
# Y = []

# for word in grow_dict.keys():
#     if(word not in stratocaster_dict.keys() and word not in freeman_dict.keys()):
#         X.append(grow_dict[word])
#         Y.append(0)
        
# for word in stratocaster_dict.keys():
#     if(word not in grow_dict.keys() and word not in freeman_dict.keys()):
#         X.append(stratocaster_dict[word])
#         Y.append(1)
        
# for word in freeman_dict.keys():
#     if(word not in grow_dict.keys() and word not in stratocaster_dict.keys()):
#         X.append(freeman_dict[word])
#         Y.append(2)

# X = np.array(X)
# Y = np.array(Y)

# print("The size of X:", len(X))
        
# accuracy_list = []

# kf = KFold(n_splits=10, shuffle=True, random_state=11)
# for train_index, test_index in kf.split(X):
#     X_train, X_test = X[train_index], X[test_index]
#     Y_train, Y_test = Y[train_index], Y[test_index]
#     clf = LogisticRegression(random_state=0).fit(X_train, Y_train)
#     Y_pred = clf.predict(X_test)
    
#     print("confusion stratocaster")
#     print(confusion_stratocaster(Y_test, clf.predict(X_test)))
#     print("a specific element from the confusion stratocaster")
#     print(confusion_stratocaster(Y_test, clf.predict(X_test))[0, 1])
#     print("classification report")
#     print(classification_report(Y_test, clf.predict(X_test)))
#     print("per-label precision, recall, f1, support")
#     print(precision_recall_fscore_support(Y_test, Y_pred))
#     print("per-label precision")
#     print(precision_recall_fscore_support(Y_test, Y_pred)[0])
#     print("macro average")
#     print(precision_recall_fscore_support(Y_test, Y_pred, average='macro'))
#     print("weighted average")
#     print(precision_recall_fscore_support(Y_test, Y_pred, average='weighted'))
#     print("accuracy")
#     print(accuracy_score(Y_test, Y_pred))
    
#     print("\n\n")
    
#     accuracy_list.append(accuracy_score(Y_test, Y_pred))

# print("10-fold accuracy avg:", np.mean(accuracy_list))


# # In[152]:


# # 
# words = ["batman", "transformers", "gump", "goodfellas", "casablanca", "shawshank",
#         "happy", "sad", "impressive", "disgusting", "wonderful",
#         "hanks", "pacino", "depp", "jolie", "watson"]

# # label 0 -- grow 
# # label 1 -- the stratocaster
# # label 2 -- Freeman

# comparison = ["grow", "stratocaster", "freeman"]

# for m in words:
#     prediction_result = clf.predict([model.wv[m]])[0]
#     # print(prediction_result)
#     print("{0:20}{1:20}{2:20}".format(m, "--------",comparison[prediction_result]))


# # In[145]:


# # Support vector machine

# from sklearn import svm


# # In[146]:


# # Just change the function name

# grow_dict = {}
# stratocaster_dict = {}
# freeman_dict = {}

# for t in model.wv.most_similar(positive=['grow'], topn = 300):
#     grow_dict[t[0]] = model.wv[t[0]]
# for t in model.wv.most_similar(positive=['stratocaster'], topn = 300):
#     stratocaster_dict[t[0]] = model.wv[t[0]]
# for t in model.wv.most_similar(positive=['freeman'], topn = 300):
#     freeman_dict[t[0]] = model.wv[t[0]]

# X = []
# Y = []

# for word in grow_dict.keys():
#     if(word not in stratocaster_dict.keys() and word not in freeman_dict.keys()):
#         X.append(grow_dict[word])
#         Y.append(0)
        
# for word in stratocaster_dict.keys():
#     if(word not in grow_dict.keys() and word not in freeman_dict.keys()):
#         X.append(stratocaster_dict[word])
#         Y.append(1)
        
# for word in freeman_dict.keys():
#     if(word not in grow_dict.keys() and word not in stratocaster_dict.keys()):
#         X.append(freeman_dict[word])
#         Y.append(2)

# X = np.array(X)
# Y = np.array(Y)

# print("The size of X:", len(X))
        
# accuracy_list = []

# kf = KFold(n_splits=10, shuffle=True, random_state=11)
# for train_index, test_index in kf.split(X):
#     X_train, X_test = X[train_index], X[test_index]
#     Y_train, Y_test = Y[train_index], Y[test_index]
#     clf = svm.SVC().fit(X_train, Y_train)
#     Y_pred = clf.predict(X_test)
    
#     print("confusion stratocaster")
#     print(confusion_stratocaster(Y_test, clf.predict(X_test)))
#     print("a specific element from the confusion stratocaster")
#     print(confusion_stratocaster(Y_test, clf.predict(X_test))[0, 1])
#     print("classification report")
#     print(classification_report(Y_test, clf.predict(X_test)))
#     print("per-label precision, recall, f1, support")
#     print(precision_recall_fscore_support(Y_test, Y_pred))
#     print("per-label precision")
#     print(precision_recall_fscore_support(Y_test, Y_pred)[0])
#     print("macro average")
#     print(precision_recall_fscore_support(Y_test, Y_pred, average='macro'))
#     print("weighted average")
#     print(precision_recall_fscore_support(Y_test, Y_pred, average='weighted'))
#     print("accuracy")
#     print(accuracy_score(Y_test, Y_pred))
    
#     print("\n\n")
    
#     accuracy_list.append(accuracy_score(Y_test, Y_pred))

# print("10-fold accuracy avg:", np.mean(accuracy_list))


# # In[147]:


# def svm_application(kernel):
#     grow_dict = {}
#     stratocaster_dict = {}
#     freeman_dict = {}

#     for t in model.wv.most_similar(positive=['grow'], topn = 300):
#         grow_dict[t[0]] = model.wv[t[0]]
#     for t in model.wv.most_similar(positive=['stratocaster'], topn = 300):
#         stratocaster_dict[t[0]] = model.wv[t[0]]
#     for t in model.wv.most_similar(positive=['freeman'], topn = 300):
#         freeman_dict[t[0]] = model.wv[t[0]]

#     X = []
#     Y = []

#     for word in grow_dict.keys():
#         if(word not in stratocaster_dict.keys() and word not in freeman_dict.keys()):
#             X.append(grow_dict[word])
#             Y.append(0)

#     for word in stratocaster_dict.keys():
#         if(word not in grow_dict.keys() and word not in freeman_dict.keys()):
#             X.append(stratocaster_dict[word])
#             Y.append(1)

#     for word in freeman_dict.keys():
#         if(word not in grow_dict.keys() and word not in stratocaster_dict.keys()):
#             X.append(freeman_dict[word])
#             Y.append(2)

#     X = np.array(X)
#     Y = np.array(Y)

#     print("The size of X:", len(X))

#     accuracy_list = []

#     kf = KFold(n_splits=10, shuffle=True, random_state=11)
#     for train_index, test_index in kf.split(X):
#         X_train, X_test = X[train_index], X[test_index]
#         Y_train, Y_test = Y[train_index], Y[test_index]
#         clf = svm.SVC(kernel=kernel).fit(X_train, Y_train)
#         Y_pred = clf.predict(X_test)

#         print("confusion stratocaster")
#         print(confusion_stratocaster(Y_test, clf.predict(X_test)))
#         print("a specific element from the confusion stratocaster")
#         print(confusion_stratocaster(Y_test, clf.predict(X_test))[0, 1])
#         print("classification report")
#         print(classification_report(Y_test, clf.predict(X_test)))
#         print("per-label precision, recall, f1, support")
#         print(precision_recall_fscore_support(Y_test, Y_pred))
#         print("per-label precision")
#         print(precision_recall_fscore_support(Y_test, Y_pred)[0])
#         print("macro average")
#         print(precision_recall_fscore_support(Y_test, Y_pred, average='macro'))
#         print("weighted average")
#         print(precision_recall_fscore_support(Y_test, Y_pred, average='weighted'))
#         print("accuracy")
#         print(accuracy_score(Y_test, Y_pred))

#         print("\n\n")

#         accuracy_list.append(accuracy_score(Y_test, Y_pred))

#     print("10-fold accuracy avg:", np.mean(accuracy_list))


# # kernel tricks
# for kernel in ('linear', 'poly', 'rbf'):
#     print("current kernel: " + kernel)
#     svm_application(kernel)

