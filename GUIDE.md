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


# Questions: 
1. The corpose of text is the statements across all debates. Each statement is a document? 
   1. Current solution: each document contains all the text on one side (e.g. pro or con) in a single debate. 
2. The accuracy of the model drops despite higher range of n-gram (73 to 63% from unigram to (1, 3) gram)
