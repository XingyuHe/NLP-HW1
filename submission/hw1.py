import argparse
import collections
from numpy.lib.function_base import vectorize
import pandas as pd 
from pandas.api.types import CategoricalDtype
import numpy as np
import matplotlib.pyplot as plt
import json
from scipy import sparse
from scipy.spatial import distance
import sklearn.metrics
import sklearn.neighbors
import sklearn.linear_model
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report, accuracy_score, balanced_accuracy_score, plot_confusion_matrix
from better_profanity import profanity
import itertools
import os 
import pickle
import joblib
import time 

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer

# ========================================= preprocessing ==============================================
def get_texts(df):
    '''
    Return a list of statements in df without differentiating the side of the speaker
    '''

    texts = []
    for round in df.loc[:, 'rounds']:
        for sub_round in round:
            for speech in sub_round:
                texts.append(speech['text'])

    return texts

def get_text_by_side(df): 
    '''
    Return a list of documents where each document contains all text on one side in a 
    single debate
    
    text = [[Pro statement 1, Pro statement 2, ... Pro statement n],
            [Con statement 1, Con statement 2, ... Con statement m]]
            where n, m is the total number of statements from Pro and Con side across
            all debates

    size: [n x 2 x # statements in each debate]
    '''

    text = []
    for round in df.loc[:, 'rounds']:
        round_text = collections.defaultdict(list)

        for sub_round in round:
            for speech in sub_round: 
                round_text[speech['side']].append(speech['text'])

        
        text.append(["".join(round_text['Pro']), "".join(round_text['Con'])])

    return np.array(text)

def get_winner(df): 
    '''
    Cons gets mapped to 0 and pro gets mapped to 1
    '''
    return np.array(df.loc[:, "winner"].replace({"Con": 0, "Pro": 1}))

# ========================================= Get user features ==============================================
class Transformer_get_any_user_align(BaseEstimator, TransformerMixin):
    def __init__(self, column, df_user):
        self.df_user = df_user
        self.category = list(set(self.df_user.loc[:, column]))
        self.religous_type = CategoricalDtype(self.category, ordered=True)
        self.column = column
        
    def fit(self, X, y=None):
        return self 
    
    def transform(self, X, y=None):
        df = X["Pro"]['df']
        voters_column = df.loc[:, 'voters'].apply(self.get_column_voters)
        
        pro_column = df.loc[:, 'pro_debater'].apply(self.get_column)
        pro_column_align = pd.DataFrame({'user_{}'.format(self.column): pro_column,
                                           'voters_{}'.format(self.column): voters_column}).apply(self.get_match_column, 
                                                                                       axis=1)
        
        con_column = df.loc[:, 'con_debater'].apply(self.get_column)
        con_column_align = pd.DataFrame({'user_{}'.format(self.column): con_column,
                                           'voters_{}'.format(self.column): voters_column}).apply(self.get_match_column,
                                                                                       axis=1)
        
        return np.hstack([np.vstack(pro_column_align.values), np.vstack(con_column_align.values)])
        
        
    def get_column(self, user): 
        user_column = self.df_user.loc[user, self.column]
        return user_column

    def get_column_voters(self, voters): 
        column_vectors = []
        voters = np.array(voters)
        eligible_voters = voters[list(map(lambda voter: voter in self.df_user.index, voters))]
        if len(eligible_voters) > 0:
            data = np.array(list(map(self.get_column, eligible_voters)))
        else:
            data = np.nan
        return data

    def get_match_column(self, row): 
        user_column, voters_column = row["user_{}".format(self.column)], row["voters_{}".format(self.column)]

        if voters_column is np.nan:
            return [0, 0, 0]

        if user_column == 'Not Saying':
            return np.array([0, 1, 0])

        feature = np.array([0, 0, 0])
        for v_r in voters_column:
            if v_r == 'Not Saying' :
                feature += np.array([0, 1, 0])
            elif v_r == user_column:
                feature += np.array([0, 0, 1])
            else:
                feature += np.array([1, 0, 0])

        return feature / np.sum(feature)

class Transformer_get_political_align(BaseEstimator, TransformerMixin):
    def __init__(self, df_user):
        self.df_user = df_user
        self.category = list(set(self.df_user.political_ideology))
        self.religous_type = CategoricalDtype(self.category, ordered=True)

        
    def fit(self, X, y=None):
        return self 
    
    def transform(self, X, y=None):
        df = X["Pro"]['df']
        voters_political = df.loc[:, 'voters'].apply(self.get_political_voters)
        
        pro_political = df.loc[:, 'pro_debater'].apply(self.get_political)
        pro_political_align = pd.DataFrame({'user_political': pro_political,
                                           'voters_political': voters_political}).apply(self.get_match_political, 
                                                                                       axis=1)
        
        con_political = df.loc[:, 'con_debater'].apply(self.get_political)
        con_political_align = pd.DataFrame({'user_political': con_political,
                                           'voters_political': voters_political}).apply(self.get_match_political,
                                                                                       axis=1)
        
        return np.hstack([np.vstack(pro_political_align.values), np.vstack(con_political_align.values)])
        
        
    def get_political(self, user): 
        user_political = self.df_user.loc[user, "political_ideology"]
        return user_political

    def get_political_voters(self, voters): 
        political_vectors = []
        voters = np.array(voters)
        eligible_voters = voters[list(map(lambda voter: voter in self.df_user.index, voters))]
        if len(eligible_voters) > 0:
            data = np.array(list(map(self.get_political, eligible_voters)))
        else:
            data = np.nan
        return data

    def get_match_political(self, row): 
        user_political, voters_political = row["user_political"], row["voters_political"]

        if voters_political is np.nan:
            return [0, 0, 0]

        if user_political == 'Not Saying':
            return np.array([0, 1, 0])

        feature = np.array([0, 0, 0])
        for v_r in voters_political:
            if v_r == 'Not Saying' :
                feature += np.array([0, 1, 0])
            elif v_r == user_political:
                feature += np.array([0, 0, 1])
            else:
                feature += np.array([1, 0, 0])

        return feature / np.sum(feature)

class Transformer_get_religious_align(BaseEstimator, TransformerMixin):
    def __init__(self, df_user):
        self.df_user = df_user
        self.category = list(set(self.df_user.religious_ideology))
        self.religous_type = CategoricalDtype(self.category, ordered=True)
        
    def fit(self, X, y=None):
        return self 
    
    def transform(self, X, y=None):
        df = X["Pro"]['df']
        voters_religious = df.loc[:, 'voters'].apply(self.get_religious_voters)
        
        pro_religious = df.loc[:, 'pro_debater'].apply(self.get_religious)
        pro_religious_align = pd.DataFrame({'user_religious': pro_religious,
                                           'voters_religious': voters_religious}).apply(self.get_match_religious, 
                                                                                       axis=1)
        
        con_religious = df.loc[:, 'con_debater'].apply(self.get_religious)
        con_religious_align = pd.DataFrame({'user_religious': con_religious,
                                           'voters_religious': voters_religious}).apply(self.get_match_religious,
                                                                                       axis=1)
        
        return np.hstack([np.vstack(pro_religious_align.values), np.vstack(con_religious_align.values)])
        
        
    def get_religious(self, user): 
        user_religious = self.df_user.loc[user, "religious_ideology"]
        return user_religious

    def get_religious_voters(self, voters): 
        religious_vectors = []
        voters = np.array(voters)
        eligible_voters = voters[list(map(lambda voter: voter in self.df_user.index, voters))]
        if len(eligible_voters) > 0:
            data = np.array(list(map(self.get_religious, eligible_voters)))
        else:
            data = np.nan
        return data

    def get_match_religious(self, row): 
        user_religious, voters_religious = row["user_religious"], row["voters_religious"]

        if voters_religious is np.nan:
            return [0, 0, 0]

        if user_religious == 'Not Saying':
            return np.array([0, 1, 0])

        feature = np.array([0, 0, 0])
        for v_r in voters_religious:
            if v_r == 'Not Saying' :
                feature += np.array([0, 1, 0])
            elif v_r == user_religious:
                feature += np.array([0, 0, 1])
            else:
                feature += np.array([1, 0, 0])

        return feature / np.sum(feature)

        



class Transformer_get_cosine_similarity(BaseEstimator, TransformerMixin):
    def __init__(self, df_user):
        self.df_user = df_user
        self.cache = {}
        
    def fit(self, X, y=None):
        return self 
    
    def transform(self, X, y=None):
        key = str(len(X['Pro']['document']))
        if key in self.cache:
            return self.cache[key]
        
        df = X["Pro"]['df']
        
        big_issues_vector_con = df.loc[:, "con_debater"].apply(self.get_big_issues_vector)
        big_issues_vector_pro = df.loc[:, "pro_debater"].apply(self.get_big_issues_vector)
        big_issues_vector_voters = df.loc[:, "voters"].apply(self.get_big_issues_vector_voters)
        big_issues_df = pd.DataFrame.from_dict({
            "pro": big_issues_vector_pro, 
            "con": big_issues_vector_con, 
            "voters": big_issues_vector_voters
        })
        similarity_df = big_issues_df.apply(self.get_cosine_similarity_voters, axis=1)
        
        similarity_feature = np.hstack([similarity_df[['avg_similarity_pro']].values, 
                                        similarity_df[['avg_similarity_con']].values])
        self.cache[key] = similarity_feature
        return similarity_feature
        

    def get_big_issues_vector(self, user): 
        if user not in self.df_user.index:
            return np.nan
        user_big_issues_dict = self.df_user.loc[user, "big_issues_dict"]
        opinion_type = CategoricalDtype(['N/O', 'Und', 'Pro', 'N/S', 'Con'], ordered=True)
        user_big_issues_df = pd.DataFrame(data={"opinion": user_big_issues_dict.values()}, 
                                          index=user_big_issues_dict.keys(),
                                          dtype=opinion_type)


        vector = np.concatenate(pd.get_dummies(user_big_issues_df).values, axis=0)
        return vector

    def get_big_issues_vector_voters(self, voters): 
        big_issues_vectors = []
        voters = np.array(voters)
        eligible_voters = voters[list(map(lambda voter: voter in self.df_user.index, voters))]
        if len(eligible_voters) > 0:
            data = np.vstack(list(map(self.get_big_issues_vector, eligible_voters)))
        else:
            data = np.nan
        return data


    def get_cosine_similarity_voters(self, row):
        con = row['con']
        pro = row['pro']
        voters = row['voters']
        
        if np.isnan(np.sum(voters)):
            return pd.Series([np.nan, np.nan, 0.5, 0.5],
                            index=["similarity_pro", "similarity_con", "avg_similarity_pro", "avg_similarity_con"])

        if con is np.nan:
            con_similarity = np.nan
            con_avg_similarity = 0.5
        else:
            con_similarity = np.array(list(map(lambda voter: distance.cosine(voter, con), voters)))
            con_avg_similarity = np.average(con_similarity)
        if pro is np.nan:
            pro_similarity = np.nan
            pro_avg_similarity = 0.5
        else:
            pro_similarity = np.array(list(map(lambda voter: distance.cosine(voter, pro), voters)))
            pro_avg_similarity = np.average(pro_similarity)
            
        return pd.Series([pro_similarity, con_similarity, pro_avg_similarity, con_avg_similarity],
                        index=["similarity_pro", "similarity_con", "avg_similarity_pro", "avg_similarity_con"])
    
# ========================================= Get linguistic features ==============================================
class Transformer_identity(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    def fit(self, X, y=None):
        return self 
    
    def transform(self, X, y=None):
        return X

class Transformer_get_length(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    def fit(self, X, y=None):
        return self 
    
    def transform(self, X, y=None): 
        '''X should have two columns. One is the document and ther other is the corresponding unigram vector'''
        # Count the number if unigrams in a feature
        unigram = X["unigram"]
        
        length = unigram.sum(axis=1)
        return length
class Transformer_get_reference_to_opponent(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    def fit(self, X, y=None):
        return self 
    
    def transform(self, X, y=None): 
        # Count the number of times the opponent's username is mentioned 
        df = X["df"]
        
        count = df.apply(self.count_opponent, axis=1).values
        count = np.reshape(count, newshape=[-1, 1])
        
        return count 
    
    def count_opponent(self, row):
        document_lower = row["document"].lower() 
        count = document_lower.count(row["opponent"]) + document_lower.count("opponent")
        
        return count
            
class Transformer_get_swear_words(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.matrix = None
    def fit(self, X, y=None):
        unigram_vectorizer = X['unigram_vectorizer']
        vector = list(map(lambda x: int(profanity.contains_profanity(x)), 
                                unigram_vectorizer.get_feature_names()))

        self.matrix = np.reshape(vector, newshape=[-1, 1])
        return self 

    def transform(self, X, y=None):
    #     perhaps get rid some of the swear words because they look like they are necessary words 
    #     for discussion such as arian, sodom 
        unigram = X['unigram']
        unigram_vectorizer = X['unigram_vectorizer']
        

        swear_pro = unigram @ self.matrix
        return swear_pro

class Transformer_get_personal_pronouns(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.personal_pronouns = pd.Series(
            ["I", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them"])
        self.matrix = None
    def fit(self, X, y=None):
        return self 
    def transform(self, X, y=None):

        document = X['df'].loc[:, 'document']

        all_counts = []

        for name in self.personal_pronouns:
            count = np.array(list(map(lambda x: x.count(" {} ".format(name)), document)))
            count = np.reshape(count, newshape=[-1, 1])
            all_counts.append(count)

        personal_pronouns_feature = np.hstack(all_counts)
        return personal_pronouns_feature

        
class Transformer_get_questions(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        document = X['df']['document']

        question_count = np.array(list(map(lambda x: x.count("?"), document)))
        question_count = np.reshape(question_count, newshape=[-1, 1])
        return question_count


class Transformer_get_websites(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        document = X['df']['document']
        website_count = np.array(list(map(lambda x: x.count("http"), document)))
        website_count = np.reshape(website_count, newshape=[-1, 1])
        return website_count

class Transformer_get_exclamation(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):

        document = X['df']['document']
        exclamation_count = np.array(list(map(lambda x: x.count("!"), document)))
        exclamation_count = np.reshape(exclamation_count, newshape=[-1, 1])
        return exclamation_count

class Transformer_get_number(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.matrix_number = None
        pass
    def fit(self, X, y=None):
        unigram_vectorizer = X['unigram_vectorizer']
        vector_number = list(map(lambda x: int(x[0].isnumeric()), unigram_vectorizer.get_feature_names()))
        self.matrix_number = np.reshape(vector_number, newshape=[-1, 1])
        return self
    def transform(self, X, y=None):
        document = X['df']['document']
        unigram = X['unigram']
        number = unigram @ self.matrix_number

        return number

class Transformer_get_modal_verb(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.matrix_modal_verb = None
        pass
    def fit(self, X, y=None):
        unigram_vectorizer = X['unigram_vectorizer']
        modal_verbs = set(["can", "could", "may", "might", "shall", "should", "will", "would", "must"])
        vector_modal_verb = list(map(lambda x: int(x in modal_verbs), unigram_vectorizer.get_feature_names()))
        self.matrix_modal_verb = np.reshape(vector_modal_verb, newshape=[-1, 1])
        return self
    
    def transform(self, X, y=None):

        document = X['df']['document']
        unigram = X['unigram']
        modal_verb = unigram @ self.matrix_modal_verb

        return modal_verb

# ========================================= Get Lexicon features ==============================================
class Transformer_get_vad_feature(BaseEstimator, TransformerMixin):
    def __init__(self, df_vad):
        self.df_vad = df_vad
        
    def fit(self, X, y=None):
        unigram_vectorizer = X['unigram_vectorizer']
        
        word_vad = self.df_vad.index
        word_vector_vad = unigram_vectorizer.transform(word_vad)
        self.matrix_vad = word_vector_vad.T @ self.df_vad 
        return self 
    
    def transform(self, X, y=None):
        gram_pro = X['unigram']
        feature_pro = gram_pro @ self.matrix_vad / np.sum(gram_pro)
        return feature_pro
        
class Transformer_get_connotation_feature(BaseEstimator, TransformerMixin):
    def __init__(self, df_connotation):
        self.connotation = df_connotation
    
    def fit(self, X, y=None):
        unigram_vectorizer = X['unigram_vectorizer']
        
        word_connotation = self.df_connotation.index
        word_vector_connotation = unigram_vectorizer.transform(word_connotation)
        self.matrix_connotation = word_vector_connotation.T @ self.df_connotation
        return self 
    
    def transform(self, X, y=None):
        gram_pro = X['unigram']
        feature_pro = gram_pro @ self.matrix_connotation / np.sum(gram_pro)
        return feature_pro
    
# ========================================= Get ngrams features ==============================================
class Transformer_get_trigrams(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.existing_trigram = {}
    def fit(self, X, y=None):
        return self 
    def transform(self, X, y=None):
        key = X['side'] + str(len(X['df']['document']))
        if key in self.existing_trigram:
            return self.existing_trigram[key]
        
        document = X['df']['document'] 
        trigram_vectorizer = X['trigram_vectorizer']
        
        trigram = trigram_vectorizer.transform(document)
        self.existing_trigram[key] = trigram
        
        return trigram

class Transformer_get_unigrams(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    def fit(self, X, y=None):
        return self 
    def transform(self, X, y=None):
        document = X['document'] 
        df = X['df'] 
        side = X['side']
        opponent = 'pro_debater' if side == 'Con' else 'con_debater'
        unigram_vectorizer = X['unigram_vectorizer']
        trigram_vectorizer = X['trigram_vectorizer']
        
        df_side = pd.DataFrame.from_dict(
            {
                "document": document,
                "opponent": df.loc[:, opponent]
            }
        )

        feature = {"df": df_side, 
                    "unigram": unigram_vectorizer.transform(document),
                    "unigram_vectorizer": unigram_vectorizer,
                    "trigram_vectorizer": trigram_vectorizer,
                    "side": side
                    }
        
        return feature

# ========================================= Preprocessing ==============================================
class Transformer_separate_document(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.unigram_vectorizer = None
        self.trigram_vectorizer = None
        self.document_side = None
        self.document = None
        
    def fit(self, df, y=None):
        document_side = get_text_by_side(df)
        document = [side[0] + side[1] for side in document_side]
        
        if not self.unigram_vectorizer:
            self.unigram_vectorizer = CountVectorizer()
            self.unigram_vectorizer.fit(document)
            
        if not self.trigram_vectorizer:
            self.trigram_vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.8, 
                                                 min_df=0.2, stop_words='english', ngram_range=(1,3))
            self.trigram_vectorizer.fit(document)
            
        return self 
    
    def transform(self, df, y=None):
        document_side = get_text_by_side(df)
        document = [side[0] + side[1] for side in document_side]
        
        return {"Pro": 
                        {"df": df,
                        "unigram_vectorizer": self.unigram_vectorizer,
                        "trigram_vectorizer": self.trigram_vectorizer,
                        "document": document_side[:, 0],
                        "side": "Pro"},
               "Con": 
                        {"df": df,
                        "unigram_vectorizer": self.unigram_vectorizer,
                        "trigram_vectorizer": self.trigram_vectorizer,
                        "document": document_side[:, 1],
                        "side": "Con"}
               }

# Using only ngram 
# Using only ngram + lexicon 
# Using only ngram + linguistic 
# Using only ngram + users

# Using only ngram + lexicon + linguistic
# Using only ngram + lexicon + linguistic + users 

def get_transformers(transformer_path, df_connotation, df_vad, df_users):

    # if os.path.isfile(transformer_path):

    #     with open(transformer_path, 'rb') as transformer_file:
    #         all_transformer_dict = pickle.load(transformer_file)

    #     return list(map(lambda name: all_transformer_dict[name],
    #                     ['ngram', 'lexicon', 'linguistic', 'users', 'other']))

    transformer_separate_document = Transformer_separate_document()
    transformer_get_pro = FunctionTransformer(lambda both_side_X: both_side_X['Pro'])
    transformer_get_con = FunctionTransformer(lambda both_side_X: both_side_X['Con'])
    
    transformer_get_length = Transformer_get_length()
    transformer_get_reference_to_opponent = Transformer_get_reference_to_opponent()
    transformer_get_swear_words = Transformer_get_swear_words()
    transformer_get_personal_pronouns = Transformer_get_personal_pronouns()

    transformer_get_questions = Transformer_get_questions()
    transformer_get_websites = Transformer_get_websites()
    transformer_get_exclamation = Transformer_get_exclamation()
    transformer_get_number = Transformer_get_number()
    transformer_get_modal_verb = Transformer_get_modal_verb()

    transformer_get_unigrams = Transformer_get_unigrams()
    transformer_get_trigrams = Transformer_get_trigrams()

    transformer_get_connotation_feature = Transformer_get_connotation_feature(df_connotation)
    transformer_get_vad_feature = Transformer_get_vad_feature(df_vad)

    transformer_get_cosine_similarity = Transformer_get_cosine_similarity(df_users)
    transformer_get_religious_align = Transformer_get_religious_align(df_users)
    transformer_get_political_align = Transformer_get_political_align(df_users)
    transformer_get_education_user_align = Transformer_get_any_user_align('education', df_users)
    transformer_get_party_user_align = Transformer_get_any_user_align('party', df_users)
    transformer_get_gender_user_align = Transformer_get_any_user_align('gender', df_users)
    transformer_get_ethnicity_user_align = Transformer_get_any_user_align('ethnicity', df_users)
    transformer_get_relationship_user_align = Transformer_get_any_user_align('relationship', df_users)


    linguistic_transformer_dict = {
            'length': transformer_get_length,
            'reference_to_opponent': transformer_get_reference_to_opponent,
            'swear_words': transformer_get_swear_words,
            'personal_pronouns': transformer_get_personal_pronouns, 
            "questions":  transformer_get_questions,
            "websites":  transformer_get_websites,
            "exclamation":  transformer_get_exclamation,
            "number":  transformer_get_number,
            "modal_verb":   transformer_get_modal_verb
    }

    lexicon_transformer_dict = {
            'connotation': transformer_get_connotation_feature,
            'vad': transformer_get_vad_feature
    }

    ngram_transformer_dict = {
            'trigram': transformer_get_trigrams,
            'unigram': transformer_get_unigrams
    }

    users_transformer_dict = {
            'cosine_similarity '         : transformer_get_cosine_similarity ,
            'religious_align'         : transformer_get_religious_align,
            'political_align'         : transformer_get_political_align,
            'education_user_align'         : transformer_get_education_user_align,
            'party_user_align'         : transformer_get_party_user_align,
            'gender_user_align'         : transformer_get_gender_user_align,
            'ethnicity_user_align'         : transformer_get_ethnicity_user_align,
            'relationship_user_align'         : transformer_get_relationship_user_align

    }

    other_transformer_dict = {
            "get_pro" : transformer_get_pro,
            "get_con" : transformer_get_con,
            'separate_document': transformer_separate_document
    }

    all_transformer_dict = {
            'ngram': ngram_transformer_dict,
            'lexicon': lexicon_transformer_dict,
            'linguistic': linguistic_transformer_dict,
            'users': users_transformer_dict,
            'other': other_transformer_dict
    }

    # with open(transformer_path, 'wb') as transformer_file:
    #     joblib.dump(all_transformer_dict, transformer_file)

    return list(map(lambda name: all_transformer_dict[name],
                    ['ngram', 'lexicon', 'linguistic', 'users', 'other']))



def get_pipeline(ngram_trans_names, lexicon_trans_names, linguistic_trans_names, users_trans_names, 
                 all_transformer_dict):

    ngram_transformer_dict, lexicon_transformer_dict, \
    linguistic_transformer_dict, users_transformer_dict, \
    other_transformer_dict                                      = all_transformer_dict
    
    def get_trans_from_name(trans_names, trans_dict): 
        return [(name, trans_dict[name]) for name in trans_names]

    linguistic_trans = get_trans_from_name(linguistic_trans_names, 
                                           linguistic_transformer_dict)
    lexicon_trans = get_trans_from_name(lexicon_trans_names, lexicon_transformer_dict)
    ngram_trans = get_trans_from_name(ngram_trans_names, ngram_transformer_dict)
    users_trans = get_trans_from_name(users_trans_names, users_transformer_dict)

    transformer_get_unigrams = ngram_transformer_dict['unigram']
    transformer_get_pro = other_transformer_dict['get_pro']
    transformer_get_con = other_transformer_dict['get_con']
    transformer_separate_document = other_transformer_dict['separate_document']

    linguistic_lexicon_trans = FeatureUnion(
        linguistic_trans + lexicon_trans + ngram_trans
    )

    side_trans = Pipeline(
        [
            ('ngram', transformer_get_unigrams), 
            ('linguistic_lexicon', linguistic_lexicon_trans)
        ]
    )

    pro_trans = Pipeline(
        [
            ('get_pro', transformer_get_pro),
            ('side_trans', side_trans) 
        ]
    )

    con_trans = Pipeline(
        [
            ('get_con', transformer_get_con),
            ('side_trans', side_trans) 
        ]
    )

    both_trans = FeatureUnion(
        [
            ('pro_feature', pro_trans),
            ('con_feature', con_trans)
        ] + users_trans
    #         ('cosine_similarity', transformer_get_cosine_similarity)
    )

    big_trans = Pipeline(
        [
            ('separate_document', transformer_separate_document), 
            ('get_both_features', both_trans),
            ('logistic_regression', sklearn.linear_model.LogisticRegression())
        ]
    )

    return big_trans


if __name__ == '__main__':
    start_time = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument('--train', dest='train', required=True,
                        help='Full path to the training file')
    parser.add_argument('--test', dest='test', required=True,
                        help='Full path to the evaluation file')
    parser.add_argument('--user_data', dest='user_data', required=True,
                        help='Full path to the user data file')
    parser.add_argument('--model', dest='model', required=True,
                        choices=["Ngram", "Ngram+Lex", "Ngram+Lex+Ling", "Ngram+Lex+Ling+User"],
                        help='The name of the model to train and evaluate.')
    parser.add_argument('--lexicon_path', dest='lexicon_path', required=True,
                        help='The full path to the directory containing the lexica.'
                             ' The last folder of this path should be "lexica".')
    parser.add_argument('--outfile', dest='outfile', required=True,
                        help='Full path to the file we will write the model predictions')
    args = parser.parse_args()

    # Remember to change this ************************************************************************8
    TRANSFORMER_PATH = os.path.join(os.getcwd(), 'transformer.pickle')

    # loading user  data 
    USER_DATA = args.user_data 
    print(USER_DATA)
    df_users = pd.read_json(USER_DATA, orient="index")

    # loading training data .jsonl
    TRAINING_DATA = args.train
    df_train = pd.read_json(TRAINING_DATA, lines=True)

    # loading validation data .jsonl
    VAL_DATA = args.test
    df_val = pd.read_json(VAL_DATA, lines=True)
    
    NRC_LEXICON_VAD = os.path.join(args.lexicon_path, "NRC-VAD-Lexicon-Aug2018Release/NRC-VAD-Lexicon.txt")
    
    df_vad = pd.read_csv(NRC_LEXICON_VAD, sep="	", header=None)
    df_vad.columns = ["word", "valence", "arousal", "dominance"]
    df_vad = df_vad.dropna()
    df_vad = df_vad.set_index("word")
    df_vad["valence"] = df_vad["valence"].astype('category')
    df_vad["arousal"] = df_vad["arousal"].astype('category')
    df_vad["dominance"] = df_vad["dominance"].astype('category')

    CONNOTATION = os.path.join(args.lexicon_path, "connotation_lexicon_a.0.1.csv")

    df_connotation = pd.read_csv(CONNOTATION, sep=",|_", header=None)
    df_connotation.columns = ["word", "pos", "connotation"] # word, part of speech, connotation
    df_connotation = df_connotation.dropna() # There are five words in the connotation that are nan 
    df_connotation = df_connotation.set_index("word")
    df_connotation["pos"] = df_connotation["pos"].astype('category')
    df_connotation = df_connotation.drop(columns=["pos"]) # drop the part of speech classification because we can't use it now 
    df_connotation["connotation"] = df_connotation["connotation"].astype('category')
    df_connotation = pd.get_dummies(df_connotation)

    model_choices = ["Ngram", "Ngram+Lex", "Ngram+Lex+Ling", "Ngram+Lex+Ling+User"]

    if args.model == model_choices[0]:
        ngram_trans_names =         ['trigram']
        lexicon_trans_names =       []
        linguistic_trans_names =    []
        users_trans_names =         [] 
    elif args.model == model_choices[1]:
        ngram_trans_names =         ['trigram']
        lexicon_trans_names =       ['vad']
        linguistic_trans_names =    []
        users_trans_names =         [] 
    elif args.model == model_choices[2]:
        ngram_trans_names =         ['trigram']
        lexicon_trans_names =       ['vad']
        linguistic_trans_names =    ['personal_pronouns', 'swear_words']
        users_trans_names =         []
    elif args.model == model_choices[3]:
        ngram_trans_names =         ['trigram']
        lexicon_trans_names =       ['vad']
        linguistic_trans_names =    ['reference_to_opponent', 'swear_words']
        users_trans_names =         ['political_align', 'gender_user_align']
    else:
        raise ValueError('The model name is not right! ')

    all_transformer_dict = get_transformers(TRANSFORMER_PATH, df_connotation, df_vad, df_users)
    clf = get_pipeline(ngram_trans_names, lexicon_trans_names, linguistic_trans_names, users_trans_names,
                        all_transformer_dict)

    clf.fit(df_train, get_winner(df_train))

    y_hat_val = clf.predict(df_val)

    with open(args.outfile, 'w') as output_file: 
        for pred in y_hat_val:
            if pred == 1: 
                output_file.write('Pro\n')
            else:
                output_file.write('Con\n')

        

    end_time = time.time()
    print("=======================================")
    print(" training mode {} took {} seconds ".format(args.model, end_time - start_time))
    print(classification_report(get_winner(df_val), 
                                y_hat_val))
    print()