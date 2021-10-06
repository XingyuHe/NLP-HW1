import argparse
import json
import itertools
import os 
import time 

from sklearn.metrics import classification_report, accuracy_score, balanced_accuracy_score, plot_confusion_matrix
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer
import sklearn.metrics
import sklearn.linear_model

from features import * 

def get_transformers(df_connotation, df_vad, df_users):

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

    all_transformer_dict = get_transformers(df_connotation, df_vad, df_users)
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