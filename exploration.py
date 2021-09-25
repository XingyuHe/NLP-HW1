#%% 
import collections
from numpy.lib.function_base import vectorize
import pandas as pd 
import numpy as np
import json
from scipy import sparse
import sklearn.metrics
import sklearn.neighbors
import sklearn.linear_model
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets import fetch_20newsgroups
# %%
# load example data
data_train = fetch_20newsgroups(subset='train', shuffle=True)
# this is just a list of strings
data_test = fetch_20newsgroups(subset='test', shuffle=True)

# %%
def get_jsonl(path):

    with open(path) as json_file:
        json_list = list(json_file)

    data_list = []
    for json_str in json_list:
        data_list.append(json.loads(json_str))

    return pd.DataFrame(data_list)
# %%
# loading user  data 
USER_DATA = './resources/data/users.json'
df_user = pd.read_json(USER_DATA, orient="index")

# loading training data .jsonl
TRAINING_DATA = './resources/data/train.jsonl'
VAL_DATA = './resources/data/val.jsonl'

df_train, df_val = get_jsonl(TRAINING_DATA), get_jsonl(VAL_DATA)
# %%
df_train.columns
df_train
# %%
# Explore the structure of rounds
one_round = df_train.loc[0, "rounds"] # this is a list of list of dictionary
two_sides = one_round[2] # this is a list consists of two sides speaking
print(speech)
speech = two_sides[0] # this is a dictionary with side (pro or con) and text
two_sides
len(two_sides)

# %%
# Creating 2 lists of speech for training set and test set 
# TODO: the shape of training texts does not match the shape of the label :(
def get_texts(df):

    texts = []
    for round in df.loc[:, 'rounds']:
        for sub_round in round:
            
            if isinstance(sub_round, dict):
                texts.append(sub_round['text'])
            
            elif isinstance(sub_round, list):
                for speech in sub_round:
                    texts.append(speech['text'])

    return texts

def get_winner(df): 
    '''
    Cons gets mapped to 0 and pro gets mapped to 1
    '''
    return df_train.loc[:, "winner"].replace({"Con": 0, "Pro": 1})


# %%
# Extracting texts from training and testing data
text_train = get_texts(df_train)
lable_train = get_winner(df_train)
text_val = get_texts(df_val)
lable_val = get_winner(df_val)
# Vectorization
vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.9, stop_words='english')
X_train = vectorizer.fit_transform(text_train)
X_val = vectorizer.transform(text_val)
y_train = np.array(lable_train)
y_val = np.array(lable_val)

X_train = sparse.csr_matrix(X_train)
X_val = sparse.csr_matrix(X_val)

# %%
# Building the model
clf = sklearn.linear_model.LogisticRegression()
clf.fit(X_train, y_train)
# %%