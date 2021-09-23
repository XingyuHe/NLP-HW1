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
   2. N-gram + 2 lexicon-based features
   3. N-gram + 2 lexicon-based features + 2 linquistic features
   4. N-gram + 2 lexicon-based features + 2 linquistic features + 2 user features
2. use logistic classifier that gives the highest accuracy on the development set
3. need 2 separate models for each of the settings aboce to measure the impact of debate topic
4. tune the model if it is tunable -> avoid overfitting
   1. look at scikit learn documentation
5. QUESTION CHECK FOR MON OFFICE HR: In order to achieve good performance, you mayexperiment with choosing thekbest features (e.g.,kbestn-grams), with differentvalues fork






