1. Programming portion: Implement classifiers for predicting debate owners 
2. Analyze features and n-gram smooting methods

# Part 1 
Goal: predict whether PRO or CON debator was more successful
Input: 
1. training and development files
   1. debate texts 
   2. information about the debaters and audience
2. skeleton code
Output: 
1. predict the winner of the debate
   1. y in {PRO, CON} indicating the winner of the debate


Questions: 
1. What sets of features are useful for predicting ebates on religious topoics vs. other topics
   1. Create experimental designs to understand the imact of these factors for these two groups of debates
   2. think about ways to partition your data to help answer the research question
   3. Use different models for each of the settings to measure the impact of debate topic

Implementations: 
1. Develop language models
   1. N-gram
   2. N-gram + lexicon-based features
   3. N-gram + lexicon-based features + 2 linquistic features
   4. N-gram + lexicon-based features + 2 linquistic features + 2 user features
2. use logistic classifier that gives the highest accuracy on the development set
3. need 2 separate models for each of the settings aboce to measure the impact of debate topic
4. tune the model if it is tunable -> avoid overfitting
   1. look at scikit learn documentation
5. QUESTION CHECK FOR MON OFFICE HR: In order to achieve good performance, you mayexperiment with choosing thekbest features (e.g.,kbestn-grams), with differentvalues fork



Notes on the tutorial: 
when shuffled, use random_state to shuffle the same way
TfidVectorizer - same vectorizer as hw0 

QUESTION: SHOULD I SHUFFLE BETWEEN VAL AND TRAIN DATA? 

the most challenging part of hw1 is to turn data into features
how can you make sense of that JSON file

exp design: 
test how a feature can help with the model
when you get to each one

for different feature combinations, on relgious topic 
for different relgious topic, these combinations of feature matter more

should you run all your features? 
use accuracy 

think about which sets of features are the best for different types of predictions


# PROGRESS REPORT:

## Choosing text corpose
N-gram takes a corporse of texts regardless of the structure of the texts. It converts each sentence into a vector, which we can use as features for prediction. We decide to use the texts across multiple debates as our text corpose. 

## Structure of the debates 
Each debate can be described by rounds of arguments between the pro and the con side. It is a list with seven lists embedded. Here is the content of the "round" column of the first debate:

1. The pro side provides an opening statement and the con side simply accepts. 
2. The pro side provides argument 1 and then the con side provides rebuttle 1. 
3. Empty list
4. The pro side provides argument 2 and the con side provides rebuttle 2. 
5. Empty list
6. The pro side provides argument 3 and the con side provides rebuttle 3. 
7. The pro sidelosing remark and the con side accepts the remark and concludes the debate

Each debate can have varying number of sub-rounds (e.g. 3, 4, 6, etc). 

## Feature construction
In the example, a round consists of 1 opening statement, 3 subrounds where the pro and the con side both provide arguments, and 1 closing statement at the end. A naive approach is to have 5 features associated with each debate and each feature is the n-gram representation of the 5 statemets. 

Since each debate have different number of statements, to make the number of features consistent across debates, we collapse the pro statements and con statements into 1 single statements on each side. Doing this can ignore the sequential nature of the debate. 

Alternatively, we 

As a sanity check, there are 
1. 1,592 debates in the training set
2. 11,173 statements, which translates to 7.0 statements per round

## N-gram model
### TF-IDF: 

Motivation: we care about the words that appear with moderate frequency that might be a good indication of the relationship between words. However, words that occur too often does not provide any information on how words are associated (e.g. the joint probability distribution) such as the, good, a, etc. To adjust for the discrepancy, we need a smart way to weight the occurence of words. 

TF is the term frequency of word t in document d defiend as tf(t, d) = log10(count(t, d) + 1).

IDF is the inverse document frequency defiend as N / (df_t) where N is the total number of documents in the collection, and df_t is he number of documents in which term t occurs. 

tf-idf weighted value w(t, d) = tf(t, d) x idf_t. 

### Defining documents: 
The corpose of text is the statements across all debates. Each statement is a document? 

Current solution: each document contains all the text on one side (e.g. pro or con) in a single debate. 


There is an error. The side on a debate is 
"side": "1. In the comment section, Anti-atheist requested that I post my arguments in round 1. However,  I did not see that until after I posted my acceptance",

# TODO:
1. Tune the ngram model but what is tunable? There is no smoothing methods here
2. Use chi-squared selection to pick out the relevant ngrams
   1. 

You should look closely at the training and validation data you are provided, and think about how
you would partition the data to help you answer this research question. Note that
you may need two separate models for each of the settings to measure the impact
of debate topic

I need two separate models for each setting. 
Religious topic: 
Non-religious topic:

Intuition: why do i need two different models fro religous topic and non-religious topic? 
We want to distriubiton of words (ngram features, linguistic features, user features, etc) 
matter conditionally based on the topic of the debate. For example, invoking certain 
expressions that appeal to pathos might matter much more for religous topics than otherwise. 

On the contrary, appealing to emotions might matter much less for non-religious debates. 
The root cause may be that religious topics might rely on a completely different set of 
axioms for their arguments. 

One way of achieving this is to create two n-gram models. One n-gram model outputs features
for religious topics and another n-gram model outputs features for non-religious topics.
By limiting the corpus within their topics, the Tf_idf scores may better reflect the 
proper weighting. For example, certain words that might only appear in winning relgious debates
but also appear in all other losing debates may now have a significantly different score from 
words that appear in only losing religous debates but appear in all other winning debates. 
Previously, these two sets of words would have similar tf_idf score but are not helpful 
towards predicting winning debates because their prediciton power within relgious topic is
diluted by the non-religous topics. By limiting the corpus scope, we can see that these 
words become helpful in both religous and non-relgious debates.

Another reason for this intuition is that the number of religious debates and non-religious debates
are imbalanced. There are 370 religious debates in the training set and there are 1222 
religious debates. There are 93 religious debates and 306 non-religious debates in the 
validation set. 

RESULT: ngram featurse do not matter with respect to training and validation accuracy. 
Partitioning the text corpus does not improve the effectiveness of the ngram features

TODO:
1. Take adventage of the sequential nature of the debate
   1. Define a document as the a continuous speech within a debate (i.e. there could be multiple) speech in a single debates
   2. Each document can have a ngram feature
   3. Each ngram feature 

FINISHED:
1. Two Tfidfvectorizer model separeately trained on religious topic and nonreligious topic: 
   1. Define a Tfidfvectorizer for both religous and non-religious topics
   2. Train the vectorizer using their respective subsets
   3. Depending the topic of the new data, we should use the two models conditionally 

# Questions: 
1. The corpose of text is the statements across all debates. Each statement is a document? 
   1. Current solution: each document contains all the text on one side (e.g. pro or con) in a single debate. 
2. The accuracy of the model drops despite higher range of n-gram (73 to 63% from unigram to (1, 3) gram)
3. How do I utilize the information on the judge of the debate? 


## Lexicon features
### Connotation lexicon
Goal: draw nuanced, connatative sentiments from objective words such as "intelligence", "human",
and "cheesecake". Algorithms encodes a diverse set of linguistic insights and prior knowledge into
a connotation lexicon

Each word has a positive or negative connotation

### NRC-VAD Lexicon
Valence, arousal, and dominancethe 
- valence is the positive--negative or pleasure--displeasure dimension; 
- arousal is the excited--calm or active--passive dimension; and 
- dominance is the powerful--weak or 'have control'--'have no control' dimension.
values are between 0 and 1 

TODO: 
1. There is a lot of words in the vectorizer that doesn't make sense such as 000000001 
   1. we need a way to get rid of it 
2. HOw to use the lexicon features 
   1. Perhaps consider the ratio between the counts and the length of the sentence (normalize it) 
   2. Perhaps consider the difference between two scores 
   3. Perhaps the neutral words are not that helpful, maybe we can convert raw count oif 
      features into percentage points   
3. linguistic features: 
   perhaps get rid some of the swear words because they look like they are necessary words 
   for discussion such as arian, sodom 

4. How do I manage mispellings
5. TFIDF model what to do if out of vocab 
6. Pipeline object 
select ngram featues 

FINISHED:
1. raw counts of each connotations in a sentence such as "positive", "negative", "neutral" 
              precision    recall  f1-score   support

           0       0.77      0.78      0.78       916
           1       0.70      0.68      0.69       676

    accuracy                           0.74      1592
   macro avg       0.74      0.73      0.73      1592
weighted avg       0.74      0.74      0.74      1592

              precision    recall  f1-score   support

           0       0.71      0.85      0.78       211
           1       0.79      0.61      0.69       188

    accuracy                           0.74       399
   macro avg       0.75      0.73      0.73       399
weighted avg       0.75      0.74      0.73       399
1. percentage counts of each connotations in a sentence such as "positive", "negative", "neutral" 
              precision    recall  f1-score   support

           0       0.88      0.94      0.91       916
           1       0.91      0.82      0.86       676

    accuracy                           0.89      1592
   macro avg       0.89      0.88      0.88      1592
weighted avg       0.89      0.89      0.89      1592

              precision    recall  f1-score   support

           0       0.71      0.84      0.77       211
           1       0.77      0.61      0.68       188

    accuracy                           0.73       399
   macro avg       0.74      0.72      0.72       399
weighted avg       0.74      0.73      0.72       399
1. percentage counts of each connotations in a sentence such as "positive", "negative"
              precision    recall  f1-score   support

           0       0.88      0.94      0.91       916
           1       0.91      0.82      0.86       676

    accuracy                           0.89      1592
   macro avg       0.89      0.88      0.88      1592
weighted avg       0.89      0.89      0.89      1592

              precision    recall  f1-score   support

           0       0.70      0.84      0.77       211
           1       0.77      0.60      0.68       188

    accuracy                           0.73       399
   macro avg       0.74      0.72      0.72       399
weighted avg       0.74      0.73      0.72       399

1. log counts of each connotations in a sentence such as "positive", "negative", "neutral" 
              precision    recall  f1-score   support

           0       0.87      0.93      0.90       916
           1       0.89      0.81      0.85       676

    accuracy                           0.88      1592
   macro avg       0.88      0.87      0.87      1592
weighted avg       0.88      0.88      0.88      1592

              precision    recall  f1-score   support

           0       0.70      0.85      0.77       211
           1       0.78      0.60      0.68       188

    accuracy                           0.73       399
   macro avg       0.74      0.72      0.72       399
weighted avg       0.74      0.73      0.73       399

1. How to design experiments: compare results from these models 
   1. Use two models - user id
   2. Use a model that include debate category - user id 
   3. Use a base model

2. Selecting features on ngrams

Either invent examples or find something in the debates 


MODEL RECORD: 
