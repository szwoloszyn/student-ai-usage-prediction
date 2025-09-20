# student-ai-usage-prediction


## Table of contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Approach](#approach)
- [Results](#results)
- [Summary](#summary)

## Overview
a ML model fitted to predict whether a student comes back  to using AI as help with homework/projects after a single session (marked as `UsedAgain` label in my dataset).

For data management and visualization, I have used `Pandas`, `Matplotlib` and `Seaborn` library.
For machine learning, I have used `Scikit-learn` library.
You can see full dependencies in [requirements file](./requirements.txt).
Project purpose is strictly educational.

## Dataset
Project is based on [AI Assistant Usage in Student Life](https://www.kaggle.com/datasets/ayeshasal89/ai-assistant-usage-in-student-life-synthetic) dataset which is marked by author as [MIT license](https://www.mit.edu/~amini/LICENSE.md). Dataset contains `10 000` entries and is fully complete (meaning no NULL in any cell). 

## Approach
I have started with visualizing data in multiple ways, but none of them showed linear correlation with my label. I was especially concerned of none correlation with personal student's `1-5` rate of AI's help. Correlation matrix showed that there is linear correlation with only one feature - categorical `FinalOutcome` of the session. It seems, the better the result, the more willing the students were to use AI in the future. Because the feature has values that can be logically sorted (`Assigment completed, Idea drafted, Confused, Gave up`), I changed them into 0-4 integers. For the rest of categorical features, I used **one-hot encoding**. I also converted date of the session to number of days since first session.

At this time, because of correlation's scores, I didn't expect much from my model. After multiple tests, I decided to go with `RandomForestClassifier` in hope of finding some non-linear correlations between features. 

## Results
At first, I should point out that 71% of my labels have value `True`. My first model has overall accuracy **74%**, so it is not a good result, but considering correlation matrix's outcome, I didn't expect much more. The biggest problem is with **False Positive** results (Almost twice as many as **True Negative**). Because of this **recall** value for `False` is only `0.36`.
I presume, it is caused by overwhelming majority of `True` values in both training and test dataset.
I have tried to improve my model by tuning hyperparameters, but changes in accuracy were minor.

Last but not least, I was checking importances of each feature in my model's training process. I have tried to train it with/without major features, which I considered useless earlier. Results were a little worse, than original.
I also compared performance across folds and between training and test sets, and did not observe signs of overfitting.

My final model contains hyperparameter indicated by GridSearchCV() method and is trained on every feature I was given. 

Final result, as well as my visualizations and efforts can be seen in my [model.ipynb](./model.ipynb) file.

## Summary
Based on what I observed, I don't think there is a way to significantly improve model's performance, but maybe I am forgetting something. If you read this (I would be suprised :D ) and have any ideas, feel free to leave your thoughts in `Issues` section.