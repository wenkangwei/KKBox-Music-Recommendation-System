# TITLE: KKBox Music Recommendation System
## Team Members: Wenkang Wei, ShuLe Zhu
## Date: 11/12/2020

# Project Description
## Probelm Statement and Motivation
This should be a brief and self-contained decription of the problem that your project aims to solve and what motivated you to solve this problem.

**Background:** A recommendation system is a subclass of an information filtering system that seeks to predict the "rating" or "preference" a user would give to an item. They are primarily used in commercial applications. We are using this system to try to collect the data for "rating" or "preference" music. In the modern-day, a recommendation system is one of the most powerful tools for marketing. Because music is so popular and one of the important companion in life. Also, Its personalization features improve customer engagement and retention. We can have a very good practice on recommendation engines because the idea of recommendation engines is something we are already familiar with such as product recommendation on eBay and amazon and music suggestion on youtube. 

**The Goal of this project:** For this project, we are to construct a recommendation system to  predict the chances of a user listening to a song repetitively after the first observable listening event within a time window was triggered. If there are recurring listening event(s) triggered within a month after the user’s very first observable listening event, its target is marked 1, and 0 otherwise in the training set.


## Introduction and Description of Data
Description of relevant knowledge. Why is this problem important? Why is it challenging? Introduce the motivations for the project question and how that question was defined through preliminary EDA.

Music has a lot of help, for example, it can cultivate sentiment and reduce stress. The role of music performance is one of the most important companions in life. However, to recommend music to users is hard due to some practical problem, such as Cold Startup problem: for some new users to this community, we are hard to recommend music to them as we don't have their information as reference. Moreover, data collected from music platform could be incompleted, such as missing age, tags/genres of music and so on, which make it difficult to identify the genres of songs preferred by users. In addition, due to large volumn of users in music platform, one of challenging problems in recommendation system is the algorithm to handle millions of data. If the recommendation algorithm is slow, it may lead to poor performance of recommedation platform and lead to customer churns. Hence, designing a good music recommendation system is still a challenging problem.

In this project, we apply the KKBox 1.7GB music dataset to build a recommendation system. In the dataset, it contains the following files:

1. train.csv
    + msno: user id
    + song_id: song id
    + source_system_tab: the name of the tab where the event was triggered. System tabs are used to categorize KKBOX mobile apps functions. For example, tab my library contains functions to manipulate the local storage, and tab search contains functions relating to the search.
    + source_screen_name: name of the layout a user sees.
    + source_type: an entry point a user first plays music on mobile apps. An entry point could be an album, online-playlist, song .. etc.
    + target: this is the target variable. target=1 means there is the recurring listening event(s) triggered within a month after the user’s very first observable listening event, target=0 otherwise.
    
 
2. test.csv
    + id: row id (will be used for submission)
    + msno: user id
    + song_id: song id
    + source_system_tab: the name of the tab where the event was triggered. System tabs are used to categorize KKBOX mobile apps functions. For example, tab my library contains functions to manipulate the local storage, and tab search contains functions relating to search.
    + source_screen_name: name of the layout a user sees.
    + source_type: an entry point a user first plays music on mobile apps. An entry point could be album, online-playlist, song .. etc.

3. sample_submission.csv
    + sample submission file in the format that we expect you to submit
    + id: same as id in test.csv
    + target: this is the target variable. target=1 means there are recurring listening event(s) triggered within a month after the user’s very first observable listening event, target=0 otherwise .
    
    
4. songs.csv
    The songs. Note that data is in unicode.

    + song_id
    + song_length: in ms
    + genre_ids: genre category. Some songs have multiple genres and they are separated by |
    + artist_name
    + composer
    + lyricist
    + language
    
    
5. members.csv
    user information.
    + msno
    + city
    + bd: age. Note: this column has outlier values, please use your judgement.
    + gender
    + registered_via: registration method
    + registration_init_time: format %Y%m%d
    + expiration_date: format %Y%m%d



## Literature Review/Related Work 
This could include noting any key papers, texts, or websites that you have used to develop your modeling approach, as well as what others have done on this problem in the past. You must properly credit sources.

https://www.kaggle.com/c/kkbox-music-recommendation-challenge/overview
This website gave us a big picture of our project
https://blog.statsbot.co/recommendation-system-algorithms-ba67f39ac9a3 
This website gave us a big overview picture of the recommendation system.
https://www.analyticsvidhya.com/blog/2016/06/quick-guide-build-recommendation-engine-python/
This website teaches us building a recommendation system in python using the graph lab library and traversed through the process of making a basic recommendation engine in Python using GrpahLab. 
https://medium.com/recombee-blog/machine-learning-for-recommender-systems-part-1-algorithms-evaluation-and-cold-start-6f696683d0ed
This website contains some machine learning information for creating a recommendation system.


## Interim Results
In the original plan, we plan to do the following steps to explore, transform the data, and construct a machine learning model to predict if users will repeat listening to songs. Finally, we are going to validate and evaluate the model.  Currently, we have loaded the KKBox Music data set from the Kaggle website and explore the data using the visualization method.
    
## EDA

In the Exploratory Data Analysis (EDA) step, we first use pandas tool kits to check the data type to determine which attributes are numerical or object attributes, then we convert the object data type to categorical data type so that we can handle the categorical data easily later. 

Then we are interested in exploring the users' behavior of listening song and choosing sources of songs, which can give us insight about the relationship between sources of song and behavior of repeating listening to those songs, so we use barplot and counterplot from seaborn to visualize some attributes, like source_type, source_screen_name, source_system_tab, song genre_ids, song composers, age,  to figure out the distributions of different attributes. 

In this step, we find that most of users tend to re-play their music from the local library or local source. Moreover, most users like listening to the song of some specific genres, which also shows that the distribution of genre_id is long-tail distribution, according to figures in our figures in Jupyter Notebook. When we are analyzing the distribution of age (bd attribution), we find there are some outliers outside the range of [0,100]. After replacing those outliers with NaN values, we find most users are around 27 years old.

We also check the number of missing values of each attribute and find that there are 11 attributes in all files that contain missing values. In this case, we simply replace the missing values in numerical data type with median value and missing values in categorical data type with the most frequent values occurring in those attributes. The visualization results can be found in our Jupiter notebook here:
https://github.com/cpsc6300/course-project-wenkang-and-shu-le/blob/main/notebooks/1_kkboxmusicrecommendation-notebook.ipynb

In the following weeks, we will tend to do visualization of bivariate to explore the relationship between features and target and then transform the data by parsing the categorical data and converting them to numerical data, using techniques like One-Hot-Encoding, label encoding, normalization so that we can use them to train the machine learning model.

# Project Progress 

|Date| Assignment| Wenkang| Shule | Completed|
|-- |--  | --  | --  | -- |
|Oct 22 |Proposal| ✓ |  ✓ | Yes|
|week 1 |Set up environment and data visualization  |✓ | ✓| Yes |
|week 2 |Data preprocessing and transformation  |   | ✓ | Ongoing |
|week 3 |Machine Learning Modeling | ✓  |   |  Ongoing |
|Nov 12| Interim Report   | ✓ | ✓| |
|week 4 |Training model and Evaluation| ✓  |   | |
|week 5 |Training model and Evaluation|  | ✓ | |
|week 6 |Wrap up and preparing website presentation  | ✓ | ✓  | |
|week 7 |Writting Final Report | ✓  | ✓ | |
|Dec 10| Website and Final Report   | ✓ | ✓ | |
|Dec 13|  Notebooks and other supporting materials  |✓  | ✓ | |

# References:
This could include the revised key papers, texts, or websites that you may use to develop your project.

https://www.kaggle.com/c/kkbox-music-recommendation-challenge/overview
https://blog.statsbot.co/recommendation-system-algorithms-ba67f39ac9a3
https://www.analyticsvidhya.com/blog/2016/06/quick-guide-build-recommendation-engine-python/
https://medium.com/recombee-blog/machine-learning-for-recommender-systems-part-1-algorithms-evaluation-and-cold-start-6f696683d0ed
