NLP HW1 

## Name
Xingyu He

## Email
xh2513@columbia.edu

## Homework number.
HW 1

## Information on how to train and test your classifier

Install required packages: 

```
pandas==1.1.5
numpy==1.19.2
matplotlib==3.3.4
scipy==1.5.2
better_profanity==0.7.0
scikit_learn==1.0
```

Running the `hw1.py` will train and test a chosen model:
```
$  python hw1.py    --train TRAIN \
                    --test TEST \
                    --user_data USER_DATA \
                    --model MODEL4\
                    --lexicon_path LEXICON_PATH \
                    --outfile OUTFILE

$  python hw1.py -h 
  --train TRAIN         Full path to the training file
  --test TEST           Full path to the evaluation file
  --user_data USER_DATA
                        Full path to the user data file
  --model {Ngram,Ngram+Lex,Ngram+Lex+Ling,Ngram+Lex+Ling+User}
                        The name of the model to train and evaluate.
  --lexicon_path LEXICON_PATH
                        The full path to the directory containing the lexica.
                        The last folder of this path should be "lexica".
  --outfile OUTFILE     Full path to the file we will write the model
```



## A description of special features (or limitations) or your classifier.
A document is defined as a string that contains all speeches made by one side of a debate. Therefore, the number of documents = the number of debates x 2. 

Ngram vectorizers are fitted on all documents from both the con and pro side. 
- unigram: fitted using CountVectorizer 
- trigram: fitted using TfIdfVectorizer
We use trigram as the input features for our supervised learning algorithm. The unigrams are inputs for constructing other featues. 

Lexicon features use the nrc-vad lexicon and the connotation lexicon. 
- vad-average: the average valence, arousal, dominance level for a debater's speeches
- connotation-percentage: get the percentage of positive, negative, and neutral words for a debater's speeches
In testing, we default to the vad-average feature. 

Linguistic features uses ngrams or raw documents.
- length: the number of words for each string in a list of string
- reference to opponent: the number of times a debater refer to its opponents by user name or with "opponent"
- swear: the number of times a person swear
- personal pronoun: the number of times a debater uses personal pronouns
- question mark: the number of times a debater uses a question mark
- website: the number of times a debater cites an external websites
- exclamation mark: the number of times a debater uses exclamation mark
- number: the number of times a debater uses a number
- modal verb: the number of times that a debater uses a modal verb

Userfeature uses users.json file which we commonly denote it as df_json 
- cosine_similarity     
For each user, for each issue, we can construct a one-hot vector that represents the user's opinion on the issue. They are PRO (in favor), CON (against), N/O (no opinion), N/S (not saying) or UND (undecided). We can horizontally concatenate vectors of a user on a single issue to obtain a representation on thier general views. We can take the larger vecters from two users and caculate their cosine similarity in order to see how different their general views are.

- political_ideology_alignment:
    a 3-dimension vector where each element represents the number of voters with different political ideologies that are not "Not Saying", the number of voters with "Not Saying" as their political ideologies, and the number voters with the same political ideologies as the debater. These numbers are then standard into percentages. Note that when the debater's political ideology is "Not Saying", all voters' political ideologies are treated as "Not Saying". 
- religious_ideology_alignment: similarly defined
- religious_alignment: similarly defined
- political_alignment: similarly defined       
- education_user_alignment: similarly defined  
- party_user_alignment: similarly defined      
- gender_user_alignment: similarly defined     
- ethnicity_user_alignment: similarly defined  
- relationship_user_alignment: similarly defined


Ngram, lexicon, linguistic, and user features are transformed on the basis of the side of the debater. For example, to caculate the unigram feature for the pro side, we aggregate all documents from the pro side in a DataFrame and uses a trained vectorizer to convert each document in a single ngram vector. 

