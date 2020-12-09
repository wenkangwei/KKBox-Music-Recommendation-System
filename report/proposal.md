# TITLE: KKBOX Music Recommendation System
## Team Members: Wenkang Wei, ShuLe Zhu
## Date: 10/22/2020

# Project Description
## Probelm Statement and Motivation
#### Problem statement
As the public is listening to all kinds of music and there is a diversity of the favors of different individuals in listening music, music users may choose different music platforms based on which platform can provide the kinds of music closest to their tastes.

In order to help music platforms figure out how to recommend suitable kinds of music and songs to individuals or groups so that they can provide better services and retain users, it is necessary to analyze tastes of individuals and the public in listening music and explore the chances that users may repeat listening some songs based on data analysis.

#### Motivation
The goals of this project are following:
1. analyze the correlation among features, like user and song, source type and user, etc, by using visualization, EDA methods
2. Using Extract load Transform (ELT) and data cleaning techniques to clean and transform data to feed machine learning model
3. Apply machine learning models (like logistic regression, decision tree, neural network, etc) to construct music recommendation system
4. Evaluate the recommendation results and give some suggestions on recommending music to users based on data analysis and business insight.


## Introduction and Description of Data
Description of relevant knowledge. Why is this problem important? Why is it challenging? Introduce the motivations for the project question and how that question was defined through preliminary EDA.

In our lives, the role of music is determined by people and the environment. Music has a lot of help, for example, it can cultivate sentiment and reduce stress. However, some people don't need music in their lives or live in music. People who never listen to music or harmony feel that music has a low sense of existence. But in fact, in their everyday lives, they actually have songs, ringing alarms, wind and rain. So, the role of music performance is one of the most important companions in life. I think that music is the greatest thing in human history, and it can’t even be said to be human, because there are still a lot of inspirations from nature. So, music is great and undeniable for the whole world.
Music plays a very important role to people’s lives. Because music can relax people's nerves, and make people feel relaxed and happy when they are nervous, irritable, and upset. First, people feel the emotions conveyed by music and then internalize them. Second, people can image some visual images while people are listening to music, which arouses emotions associated with those visual images. Last, music reminds people of some past experiences about themselves, and arouses emotions related to these experiences. 

Even we have largest dataset from KKBOX, Asia’s leading music streaming service, holding the world’s most comprehensive Asia-Pop music library with over 30 million tracks. The algorithms still struggle in key areas because the public’s now listening to all kinds of music. If we found a way to solve this solution, it will bring a lots impact to our society. 


# Plan
This could include the major timeline and milestones of your project as well as requirements of computing and storage resources.

#### Requirements
1. Computing Platform
    + Option 1: Palmetto Cluster
    + Option 2: Kaggle
    + Option 3: Google Colab
2. Software Packages:
    + Data Proprocessing:  pandas, numpy, sklearn
    + Visualization: seaborn, matplotlib
    + Machine Learning: Sklearn, tensorflow, pytorch, lightGBM

#### Possible Recommendation system models
1. Logisitc Regression and Softmax Regression
2. Decision Tree / Random Forest
3. K-NN
4. Neural network
5.  Matrix Factorization
6. Collaborate Filtering
7. Boosting machine



#### Timeline
|Date| Assignment| Wenkang| Shule |
|-- |--  | --  | --  |
|Oct 22 |Proposal| ✓ |  ✓ |
|week 1 |Set up environment and data visualization  |✓ | ✓|
|week 2 |Data preprocessing  |   | ✓ |
|week 3 |Modeling | ✓  |   |
|Nov 12| Interim Report   | ✓ | ✓|
|week 4 |Training model and Evaluation| ✓  |   |
|week 5 |Training model and Evaluation|  | ✓ |
|week 6 |Wrap up and preparing website presentation  | ✓ | ✓  |
|week 7 |Writting Final Report | ✓  | ✓ |
|Dec 10| Website and Final Report   | ✓ | ✓ |
|Dec 13|  Notebooks and other supporting materials  |✓  | ✓ |

# References:
[1] https://blog.statsbot.co/recommendation-system-algorithms-ba67f39ac9a3

[2] https://www.analyticsvidhya.com/blog/2016/06/quick-guide-build-recommendation-engine-python/

[3] https://www.kaggle.com/c/kkbox-music-recommendation-challenge/notebooks

[4] https://medium.com/recombee-blog/machine-learning-for-recommender-systems-part-1-algorithms-evaluation-and-cold-start-6f696683d0ed

