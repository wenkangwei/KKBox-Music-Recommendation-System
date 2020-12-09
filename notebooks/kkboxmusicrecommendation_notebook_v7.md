
# CPSC6300 - Final Project: KKBox-Music Recommendation System
# Author: Wenkang Wei, ShuLe Zhu

# Outline of KKBox-Music Recommendation System 
Here is an outline of the project, the project will discuss following sections
## Motivation
A recommendation system is a subclass of an information filtering system that seeks to predict the "rating" or "preference" a user would give to an item.
In Music Recommendation System, due to the fact that there are millions of musics, composers, artist and other information, it is hard for users to pick the musics they like from millions of songs and remember which songs they like and want to re-play. Hence it is necessary to build a music recommendation system to recommend musics, which users prefer and are likely to repeat listening to.

## Goal of this project
In this project, we are going to build a recommendation system to predict the chances of a user listening to a song repetitively after the first observable listening event within a time window was triggered. If there are recurring listening event(s) triggered within a month after the user’s very first observable listening event, its target is marked 1, and 0 otherwise in the training set. The same rule applies to the testing set.

## Platform
We run this notebook in Google Colab. Please read this reference to see how to download kaggle data to google colab https://www.kaggle.com/general/74235

You can run the codes with some modifications, like changing root path in Kaggle platform as well

## 0. Data Collection and Description
**Two Ways to download dataset:** Google Drive, Kaggle resource

**Descriptions of dataset:**

1. train.csv
    + msno: user id
    + song_id: song id
    + source_system_tab: the name of the tab where the event was triggered. System tabs are used to categorize KKBOX mobile apps functions. For example, tab my library contains functions to manipulate the local storage, and tab search contains functions relating to search.
    + source_screen_name: name of the layout a user sees.
    + source_type: an entry point a user first plays music on mobile apps. An entry point could be album, online-playlist, song .. etc.
    + target: this is the target variable. target=1 means there are recurring listening event(s) triggered within a month after the user’s very first observable listening event, target=0 otherwise .
    
 
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

6. song_extra_info.csv
    + song_id
    + song name - the name of the song.
    + isrc - International Standard Recording Code, theoretically can be used as an identity of a song. However, what worth to note is, ISRCs generated from providers have not been officially verified; therefore the information in ISRC, such as country code and reference year, can be misleading/incorrect. Multiple songs could share one ISRC since a single recording could be re-published several times.

## 1. Exploratory Data Analysis (EDA)
In data exploratory data analysis, we will do the following visualization and analysis to exploret the relationship between attributes:

+ Find the Description and summary of each CSV file and determine Null object, categorical attributes, numerical attributes
+ Analyze Song information
  - Plot and visualize the count of song genres, composers, artists
  - Plot and visualize the count of source types, source screen names, system tabs with target as hue axis
   
+ Analyze Member information
  - visualize the count of bd/age attribute

+ Analyze correlation between different attributes using heatmap plot

+ Plot bivariate plots to visualize and analyze relationship between attributes, like city and age/bd, expiration date and target.


## 2. Data Preprocessing 
**Note that** This section is to give some examples to preprocess data like filling missing values and removing outliers. 

**To train models directly, we start from step 3 data ETL to extract and transform data directly using integrated functions**

+ Convert String datetime data to datetime format and separate year, month, day as new features

+ Remove the outliers

+ Filling missing values

+ Transform and add new features

+ Convert features from object type into categorical type.

## 3. Data Pipeline: Extract, Transformation, Load (ETL) 
**Please Note that since RAM memory is small (only 12GB), to avoid the program crashes, You may need to directly start from Step 3 (ETL) to clean and transform data and train model later.**


## 4. Machine Learning Modeling
+ **Task**

+ **Loss Function**

+ **Evaluation Metric**  

+ **Light Gradient Boosting machines (LGBM) model**

+ **Wide and Deep Neural network model**

## 5. Model Training and validation

## 6. Model Evaluation

## 7. Report and Conclusion


.

.

# 0. Data Collection


```python
# create directory to store data
!mkdir -p ./kaggle/
```

## Method 1: Download data from my Google Drive directly (Recommended)
This method requires we use a tool called. **gdown**
If you have not install it, please install it first by command line:  

**pip install gdown**


```python
# url to dataset in Google Drive: https://drive.google.com/file/d/1-WJHZUWFtz9ksfvFoX-dc-ZKjZ6fTk0D/view?usp=sharing
! gdown --id 1-WJHZUWFtz9ksfvFoX-dc-ZKjZ6fTk0D
! unzip kkbox-dataset.zip
```

    Downloading...
    From: https://drive.google.com/uc?id=1-WJHZUWFtz9ksfvFoX-dc-ZKjZ6fTk0D
    To: /content/kkbox-dataset.zip
    362MB [00:03, 101MB/s]
    Archive:  kkbox-dataset.zip
    replace members.csv.7z? [y]es, [n]o, [A]ll, [N]one, [r]ename: n
    replace sample_submission.csv.7z? [y]es, [n]o, [A]ll, [N]one, [r]ename: N
    

## Method 2: Dowload data from kaggle website. **(Please Skip this, if you choose Method 1)**
This method requires you to download and upload API token, a file called kaggle.json from your kaggle account first.

Please refer to this tutorial https://www.kaggle.com/general/74235



```python
from google.colab import drive
drive.mount('/content/drive')
```

    Mounted at /content/drive
    


```python
!pip install kaggle
```

    Requirement already satisfied: kaggle in /usr/local/lib/python3.6/dist-packages (1.5.9)
    Requirement already satisfied: python-slugify in /usr/local/lib/python3.6/dist-packages (from kaggle) (4.0.1)
    Requirement already satisfied: slugify in /usr/local/lib/python3.6/dist-packages (from kaggle) (0.0.1)
    Requirement already satisfied: tqdm in /usr/local/lib/python3.6/dist-packages (from kaggle) (4.41.1)
    Requirement already satisfied: requests in /usr/local/lib/python3.6/dist-packages (from kaggle) (2.23.0)
    Requirement already satisfied: certifi in /usr/local/lib/python3.6/dist-packages (from kaggle) (2020.11.8)
    Requirement already satisfied: python-dateutil in /usr/local/lib/python3.6/dist-packages (from kaggle) (2.8.1)
    Requirement already satisfied: urllib3 in /usr/local/lib/python3.6/dist-packages (from kaggle) (1.24.3)
    Requirement already satisfied: six>=1.10 in /usr/local/lib/python3.6/dist-packages (from kaggle) (1.15.0)
    Requirement already satisfied: text-unidecode>=1.3 in /usr/local/lib/python3.6/dist-packages (from python-slugify->kaggle) (1.3)
    Requirement already satisfied: chardet<4,>=3.0.2 in /usr/local/lib/python3.6/dist-packages (from requests->kaggle) (3.0.4)
    Requirement already satisfied: idna<3,>=2.5 in /usr/local/lib/python3.6/dist-packages (from requests->kaggle) (2.10)
    

If using Method2 to download dataset, please make sure you download kaggle.json API token from your kaggle account to current directory.


```python
# Copy kaggle.json file to current directory

# Note: Please modify this command to upload your own kaggle.json file !

!cp /content/drive/My\ Drive/Colab\ Notebooks/KKBox-MusicRecommendationSystem/kaggle.json .


# !!chmod 600 kaggle.json
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!ls ~/.kaggle
!mkdir -p ./kaggle/
!chmod 600 /root/.kaggle/kaggle.json
!kaggle  competitions download kkbox-music-recommendation-challenge


!du -h *.csv.7z
```

    kaggle.json
    Warning: Looks like you're using an outdated API Version, please consider updating (server 1.5.10 / client 1.5.4)
    Downloading song_extra_info.csv.7z to /content
     93% 92.0M/98.8M [00:03<00:00, 16.9MB/s]
    100% 98.8M/98.8M [00:03<00:00, 27.8MB/s]
    Downloading songs.csv.7z to /content
     96% 97.0M/101M [00:05<00:00, 10.3MB/s]
    100% 101M/101M [00:05<00:00, 20.0MB/s] 
    Downloading train.csv.7z to /content
     96% 97.0M/101M [00:03<00:00, 11.0MB/s]
    100% 101M/101M [00:03<00:00, 28.2MB/s] 
    Downloading test.csv.7z to /content
     79% 33.0M/41.9M [00:02<00:01, 6.29MB/s]
    100% 41.9M/41.9M [00:02<00:00, 15.3MB/s]
    Downloading sample_submission.csv.7z to /content
      0% 0.00/453k [00:00<?, ?B/s]
    100% 453k/453k [00:00<00:00, 63.1MB/s]
    Downloading members.csv.7z to /content
      0% 0.00/1.29M [00:00<?, ?B/s]
    100% 1.29M/1.29M [00:00<00:00, 86.2MB/s]
    1.3M	members.csv.7z
    456K	sample_submission.csv.7z
    99M	song_extra_info.csv.7z
    101M	songs.csv.7z
    42M	test.csv.7z
    102M	train.csv.7z
    

## Use 7z to uncompress the csv data files


```python
!mkdir kaggle/working
!mkdir kaggle/working/train
!mkdir kaggle/working/train/data
!apt-get install p7zip
!apt-get install p7zip-full 
!7za e members.csv.7z 
!7za e songs.csv.7z 
!7za e song_extra_info.csv.7z 
!7za e train.csv.7z 
!7za e sample_submission.csv.7z 
!7za e test.csv.7z 
!mv *.csv kaggle/working/train/data
```

    Reading package lists... Done
    Building dependency tree       
    Reading state information... Done
    p7zip is already the newest version (16.02+dfsg-6).
    p7zip set to manually installed.
    0 upgraded, 0 newly installed, 0 to remove and 14 not upgraded.
    Reading package lists... Done
    Building dependency tree       
    Reading state information... Done
    p7zip-full is already the newest version (16.02+dfsg-6).
    0 upgraded, 0 newly installed, 0 to remove and 14 not upgraded.
    
    7-Zip (a) [64] 16.02 : Copyright (c) 1999-2016 Igor Pavlov : 2016-05-21
    p7zip Version 16.02 (locale=en_US.UTF-8,Utf16=on,HugeFiles=on,64 bits,2 CPUs Intel(R) Xeon(R) CPU @ 2.30GHz (306F0),ASM,AES-NI)
    
    Scanning the drive for archives:
      0M Scan         1 file, 1349856 bytes (1319 KiB)
    
    Extracting archive: members.csv.7z
    --
    Path = members.csv.7z
    Type = 7z
    Physical Size = 1349856
    Headers Size = 130
    Method = LZMA2:3m
    Solid = -
    Blocks = 1
    
      0%    Everything is Ok
    
    Size:       2503827
    Compressed: 1349856
    
    7-Zip (a) [64] 16.02 : Copyright (c) 1999-2016 Igor Pavlov : 2016-05-21
    p7zip Version 16.02 (locale=en_US.UTF-8,Utf16=on,HugeFiles=on,64 bits,2 CPUs Intel(R) Xeon(R) CPU @ 2.30GHz (306F0),ASM,AES-NI)
    
    Scanning the drive for archives:
      0M Scan         1 file, 105809525 bytes (101 MiB)
    
    Extracting archive: songs.csv.7z
    --
    Path = songs.csv.7z
    Type = 7z
    Physical Size = 105809525
    Headers Size = 122
    Method = LZMA2:24
    Solid = -
    Blocks = 1
    
      0%      2% - songs.csv                  5% - songs.csv                  9% - songs.csv                 12% - songs.csv                 15% - songs.csv                 18% - songs.csv                 20% - songs.csv                 23% - songs.csv                 26% - songs.csv                 29% - songs.csv                 32% - songs.csv                 35% - songs.csv                 37% - songs.csv                 41% - songs.csv                 43% - songs.csv                 46% - songs.csv                 49% - songs.csv                 51% - songs.csv                 54% - songs.csv                 56% - songs.csv                 59% - songs.csv                 62% - songs.csv                 65% - songs.csv                 68% - songs.csv                 70% - songs.csv                 73% - songs.csv                 76% - songs.csv                 79% - songs.csv                 82% - songs.csv                 85% - songs.csv                 87% - songs.csv                 90% - songs.csv                 92% - songs.csv                 95% - songs.csv                 98% - songs.csv                Everything is Ok
    
    Size:       221828666
    Compressed: 105809525
    
    7-Zip (a) [64] 16.02 : Copyright (c) 1999-2016 Igor Pavlov : 2016-05-21
    p7zip Version 16.02 (locale=en_US.UTF-8,Utf16=on,HugeFiles=on,64 bits,2 CPUs Intel(R) Xeon(R) CPU @ 2.30GHz (306F0),ASM,AES-NI)
    
    Scanning the drive for archives:
      0M Scan         1 file, 103608205 bytes (99 MiB)
    
    Extracting archive: song_extra_info.csv.7z
    --
    Path = song_extra_info.csv.7z
    Type = 7z
    Physical Size = 103608205
    Headers Size = 140
    Method = LZMA:25
    Solid = -
    Blocks = 1
    
      0%      1% - song_extra_info.csv                            4% - song_extra_info.csv                            6% - song_extra_info.csv                            9% - song_extra_info.csv                           11% - song_extra_info.csv                           14% - song_extra_info.csv                           17% - song_extra_info.csv                           19% - song_extra_info.csv                           22% - song_extra_info.csv                           25% - song_extra_info.csv                           27% - song_extra_info.csv                           30% - song_extra_info.csv                           33% - song_extra_info.csv                           37% - song_extra_info.csv                           39% - song_extra_info.csv                           41% - song_extra_info.csv                           44% - song_extra_info.csv                           47% - song_extra_info.csv                           50% - song_extra_info.csv                           53% - song_extra_info.csv                           55% - song_extra_info.csv                           58% - song_extra_info.csv                           61% - song_extra_info.csv                           64% - song_extra_info.csv                           67% - song_extra_info.csv                           69% - song_extra_info.csv                           72% - song_extra_info.csv                           75% - song_extra_info.csv                           78% - song_extra_info.csv                           81% - song_extra_info.csv                           83% - song_extra_info.csv                           86% - song_extra_info.csv                           89% - song_extra_info.csv                           92% - song_extra_info.csv                           96% - song_extra_info.csv                           99% - song_extra_info.csv                          Everything is Ok
    
    Size:       181010294
    Compressed: 103608205
    
    7-Zip (a) [64] 16.02 : Copyright (c) 1999-2016 Igor Pavlov : 2016-05-21
    p7zip Version 16.02 (locale=en_US.UTF-8,Utf16=on,HugeFiles=on,64 bits,2 CPUs Intel(R) Xeon(R) CPU @ 2.30GHz (306F0),ASM,AES-NI)
    
    Scanning the drive for archives:
      0M Scan         1 file, 106420688 bytes (102 MiB)
    
    Extracting archive: train.csv.7z
    --
    Path = train.csv.7z
    Type = 7z
    Physical Size = 106420688
    Headers Size = 122
    Method = LZMA2:24
    Solid = -
    Blocks = 1
    
      0%      1% - train.csv                  3% - train.csv                  5% - train.csv                  7% - train.csv                 10% - train.csv                 12% - train.csv                 15% - train.csv                 17% - train.csv                 19% - train.csv                 21% - train.csv                 23% - train.csv                 26% - train.csv                 28% - train.csv                 30% - train.csv                 32% - train.csv                 34% - train.csv                 36% - train.csv                 38% - train.csv                 41% - train.csv                 42% - train.csv                 44% - train.csv                 47% - train.csv                 48% - train.csv                 50% - train.csv                 52% - train.csv                 54% - train.csv                 56% - train.csv                 58% - train.csv                 60% - train.csv                 62% - train.csv                 64% - train.csv                 66% - train.csv                 69% - train.csv                 70% - train.csv                 72% - train.csv                 75% - train.csv                 76% - train.csv                 78% - train.csv                 80% - train.csv                 82% - train.csv                 84% - train.csv                 86% - train.csv                 88% - train.csv                 90% - train.csv                 92% - train.csv                 94% - train.csv                 96% - train.csv                 98% - train.csv                Everything is Ok
    
    Size:       971675848
    Compressed: 106420688
    
    7-Zip (a) [64] 16.02 : Copyright (c) 1999-2016 Igor Pavlov : 2016-05-21
    p7zip Version 16.02 (locale=en_US.UTF-8,Utf16=on,HugeFiles=on,64 bits,2 CPUs Intel(R) Xeon(R) CPU @ 2.30GHz (306F0),ASM,AES-NI)
    
    Scanning the drive for archives:
      0M Scan         1 file, 463688 bytes (453 KiB)
    
    Extracting archive: sample_submission.csv.7z
    --
    Path = sample_submission.csv.7z
    Type = 7z
    Physical Size = 463688
    Headers Size = 146
    Method = LZMA2:24
    Solid = -
    Blocks = 1
    
      0%    Everything is Ok
    
    Size:       29570380
    Compressed: 463688
    
    7-Zip (a) [64] 16.02 : Copyright (c) 1999-2016 Igor Pavlov : 2016-05-21
    p7zip Version 16.02 (locale=en_US.UTF-8,Utf16=on,HugeFiles=on,64 bits,2 CPUs Intel(R) Xeon(R) CPU @ 2.30GHz (306F0),ASM,AES-NI)
    
    Scanning the drive for archives:
      0M Scan         1 file, 43925208 bytes (42 MiB)
    
    Extracting archive: test.csv.7z
    --
    Path = test.csv.7z
    Type = 7z
    Physical Size = 43925208
    Headers Size = 122
    Method = LZMA2:24
    Solid = -
    Blocks = 1
    
      0%      4% - test.csv                 9% - test.csv                14% - test.csv                19% - test.csv                23% - test.csv                28% - test.csv                34% - test.csv                39% - test.csv                44% - test.csv                50% - test.csv                55% - test.csv                60% - test.csv                66% - test.csv                72% - test.csv                78% - test.csv                83% - test.csv                88% - test.csv                93% - test.csv                98% - test.csv               Everything is Ok
    
    Size:       347789925
    Compressed: 43925208
    


```python
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('./kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
```


```python
!ls ./kaggle/working/train/data/
```

    members.csv	       song_extra_info.csv  test.csv
    sample_submission.csv  songs.csv	    train.csv
    


```python
!du -h ./kaggle/working/train/data/
```

    1.7G	./kaggle/working/train/data/
    


```python
#import necessary packages here
import warnings
warnings.filterwarnings('ignore')
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from subprocess import check_output
# print(check_output(["ls", "../input"]).decode("utf8"))

np.random.seed(2020)
```

# 1. Exploratory Data Analysis


```python
root = './kaggle/working/train/data/'
# !ls ../input/kkbox-music-recommendation-challenge
train_df = pd.read_csv(root+ "train.csv")
test_df = pd.read_csv(root+ "test.csv")
song_df = pd.read_csv(root+ "songs.csv")
song_extra_df = pd.read_csv(root+ "song_extra_info.csv")
members_df = pd.read_csv(root+ "members.csv")
# sample_df = pd.read_csv(root+ "sample_submission.csv")

```

## 1.1 Take a look to the info and format of training set and testing set

### Information of training set


```python
train_df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 7377418 entries, 0 to 7377417
    Data columns (total 6 columns):
     #   Column              Dtype 
    ---  ------              ----- 
     0   msno                object
     1   song_id             object
     2   source_system_tab   object
     3   source_screen_name  object
     4   source_type         object
     5   target              int64 
    dtypes: int64(1), object(5)
    memory usage: 337.7+ MB
    


```python
train_df.count()
```




    msno                  7377418
    song_id               7377418
    source_system_tab     7352569
    source_screen_name    6962614
    source_type           7355879
    target                7377418
    dtype: int64



We can see that attributes: source_system_tab, source_screen_name, source_type contain missing values above. An example dataframe of training set is as follow:


```python
train_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>msno</th>
      <th>song_id</th>
      <th>source_system_tab</th>
      <th>source_screen_name</th>
      <th>source_type</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>FGtllVqz18RPiwJj/edr2gV78zirAiY/9SmYvia+kCg=</td>
      <td>BBzumQNXUHKdEBOB7mAJuzok+IJA1c2Ryg/yzTF6tik=</td>
      <td>explore</td>
      <td>Explore</td>
      <td>online-playlist</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Xumu+NIjS6QYVxDS4/t3SawvJ7viT9hPKXmf0RtLNx8=</td>
      <td>bhp/MpSNoqoxOIB+/l8WPqu6jldth4DIpCm3ayXnJqM=</td>
      <td>my library</td>
      <td>Local playlist more</td>
      <td>local-playlist</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Xumu+NIjS6QYVxDS4/t3SawvJ7viT9hPKXmf0RtLNx8=</td>
      <td>JNWfrrC7zNN7BdMpsISKa4Mw+xVJYNnxXh3/Epw7QgY=</td>
      <td>my library</td>
      <td>Local playlist more</td>
      <td>local-playlist</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Xumu+NIjS6QYVxDS4/t3SawvJ7viT9hPKXmf0RtLNx8=</td>
      <td>2A87tzfnJTSWqD7gIZHisolhe4DMdzkbd6LzO1KHjNs=</td>
      <td>my library</td>
      <td>Local playlist more</td>
      <td>local-playlist</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>FGtllVqz18RPiwJj/edr2gV78zirAiY/9SmYvia+kCg=</td>
      <td>3qm6XTZ6MOCU11x8FIVbAGH5l5uMkT3/ZalWG1oo2Gc=</td>
      <td>explore</td>
      <td>Explore</td>
      <td>online-playlist</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



### Information of testing set


```python
test_df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 2556790 entries, 0 to 2556789
    Data columns (total 6 columns):
     #   Column              Dtype 
    ---  ------              ----- 
     0   id                  int64 
     1   msno                object
     2   song_id             object
     3   source_system_tab   object
     4   source_screen_name  object
     5   source_type         object
    dtypes: int64(1), object(5)
    memory usage: 117.0+ MB
    


```python
test_df.count()
```




    id                    2556790
    msno                  2556790
    song_id               2556790
    source_system_tab     2548348
    source_screen_name    2393907
    source_type           2549493
    dtype: int64



In test dataset,  We can see that attributes that contain missing values: 
1. source_system_tab
2. source_screen_name
3. source_type 


```python
test_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>msno</th>
      <th>song_id</th>
      <th>source_system_tab</th>
      <th>source_screen_name</th>
      <th>source_type</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>V8ruy7SGk7tDm3zA51DPpn6qutt+vmKMBKa21dp54uM=</td>
      <td>WmHKgKMlp1lQMecNdNvDMkvIycZYHnFwDT72I5sIssc=</td>
      <td>my library</td>
      <td>Local playlist more</td>
      <td>local-library</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>V8ruy7SGk7tDm3zA51DPpn6qutt+vmKMBKa21dp54uM=</td>
      <td>y/rsZ9DC7FwK5F2PK2D5mj+aOBUJAjuu3dZ14NgE0vM=</td>
      <td>my library</td>
      <td>Local playlist more</td>
      <td>local-library</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>/uQAlrAkaczV+nWCd2sPF2ekvXPRipV7q0l+gbLuxjw=</td>
      <td>8eZLFOdGVdXBSqoAv5nsLigeH2BvKXzTQYtUM53I0k4=</td>
      <td>discover</td>
      <td>NaN</td>
      <td>song-based-playlist</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>1a6oo/iXKatxQx4eS9zTVD+KlSVaAFbTIqVvwLC1Y0k=</td>
      <td>ztCf8thYsS4YN3GcIL/bvoxLm/T5mYBVKOO4C9NiVfQ=</td>
      <td>radio</td>
      <td>Radio</td>
      <td>radio</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>1a6oo/iXKatxQx4eS9zTVD+KlSVaAFbTIqVvwLC1Y0k=</td>
      <td>MKVMpslKcQhMaFEgcEQhEfi5+RZhMYlU3eRDpySrH8Y=</td>
      <td>radio</td>
      <td>Radio</td>
      <td>radio</td>
    </tr>
  </tbody>
</table>
</div>



### Information of Song data


```python
song_df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 2296320 entries, 0 to 2296319
    Data columns (total 7 columns):
     #   Column       Dtype  
    ---  ------       -----  
     0   song_id      object 
     1   song_length  int64  
     2   genre_ids    object 
     3   artist_name  object 
     4   composer     object 
     5   lyricist     object 
     6   language     float64
    dtypes: float64(1), int64(1), object(5)
    memory usage: 122.6+ MB
    


```python
song_df.count()
```




    song_id        2296320
    song_length    2296320
    genre_ids      2202204
    artist_name    2296320
    composer       1224966
    lyricist        351052
    language       2296319
    dtype: int64



### Attributes in song data that contain missing values are following:
1. composer
2. lyricist
3. genre_ids
4. language


```python
song_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>song_id</th>
      <th>song_length</th>
      <th>genre_ids</th>
      <th>artist_name</th>
      <th>composer</th>
      <th>lyricist</th>
      <th>language</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>CXoTN1eb7AI+DntdU1vbcwGRV4SCIDxZu+YD8JP8r4E=</td>
      <td>247640</td>
      <td>465</td>
      <td>張信哲 (Jeff Chang)</td>
      <td>董貞</td>
      <td>何啟弘</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>o0kFgae9QtnYgRkVPqLJwa05zIhRlUjfF7O1tDw0ZDU=</td>
      <td>197328</td>
      <td>444</td>
      <td>BLACKPINK</td>
      <td>TEDDY|  FUTURE BOUNCE|  Bekuh BOOM</td>
      <td>TEDDY</td>
      <td>31.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>DwVvVurfpuz+XPuFvucclVQEyPqcpUkHR0ne1RQzPs0=</td>
      <td>231781</td>
      <td>465</td>
      <td>SUPER JUNIOR</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>31.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>dKMBWoZyScdxSkihKG+Vf47nc18N9q4m58+b4e7dSSE=</td>
      <td>273554</td>
      <td>465</td>
      <td>S.H.E</td>
      <td>湯小康</td>
      <td>徐世珍</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>W3bqWd3T+VeHFzHAUfARgW9AvVRaF4N5Yzm4Mr6Eo/o=</td>
      <td>140329</td>
      <td>726</td>
      <td>貴族精選</td>
      <td>Traditional</td>
      <td>Traditional</td>
      <td>52.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
print("Unique Song amount in trainset:",train_df['song_id'].nunique())
print("Unique Song amount in testset:", test_df['song_id'].nunique())
print("Unique Song amount in song list:",song_df['song_id'].nunique())
```

    Unique Song amount in trainset: 359966
    Unique Song amount in testset: 224753
    Unique Song amount in song list: 2296320
    

### Information of song extra data


```python
song_extra_df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 2295971 entries, 0 to 2295970
    Data columns (total 3 columns):
     #   Column   Dtype 
    ---  ------   ----- 
     0   song_id  object
     1   name     object
     2   isrc     object
    dtypes: object(3)
    memory usage: 52.6+ MB
    


```python
song_extra_df.count()
```




    song_id    2295971
    name       2295969
    isrc       2159423
    dtype: int64




```python
song_extra_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>song_id</th>
      <th>name</th>
      <th>isrc</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>LP7pLJoJFBvyuUwvu+oLzjT+bI+UeBPURCecJsX1jjs=</td>
      <td>我們</td>
      <td>TWUM71200043</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ClazTFnk6r0Bnuie44bocdNMM3rdlrq0bCGAsGUWcHE=</td>
      <td>Let Me Love You</td>
      <td>QMZSY1600015</td>
    </tr>
    <tr>
      <th>2</th>
      <td>u2ja/bZE3zhCGxvbbOB3zOoUjx27u40cf5g09UXMoKQ=</td>
      <td>原諒我</td>
      <td>TWA530887303</td>
    </tr>
    <tr>
      <th>3</th>
      <td>92Fqsy0+p6+RHe2EoLKjHahORHR1Kq1TBJoClW9v+Ts=</td>
      <td>Classic</td>
      <td>USSM11301446</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0QFmz/+rJy1Q56C1DuYqT9hKKqi5TUqx0sN0IwvoHrw=</td>
      <td>愛投羅網</td>
      <td>TWA471306001</td>
    </tr>
  </tbody>
</table>
</div>



### Information of member data


```python
members_df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 34403 entries, 0 to 34402
    Data columns (total 7 columns):
     #   Column                  Non-Null Count  Dtype 
    ---  ------                  --------------  ----- 
     0   msno                    34403 non-null  object
     1   city                    34403 non-null  int64 
     2   bd                      34403 non-null  int64 
     3   gender                  14501 non-null  object
     4   registered_via          34403 non-null  int64 
     5   registration_init_time  34403 non-null  int64 
     6   expiration_date         34403 non-null  int64 
    dtypes: int64(5), object(2)
    memory usage: 1.8+ MB
    

## 1.2 Explore Song information


```python
# Merge two dataframe based on song_id so that we can analyze the song information together with training data

user_music_df = train_df.merge(song_df,on='song_id',how="left", copy =False)
user_music_df["song_id"] = user_music_df["song_id"].astype("category")
user_music_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>msno</th>
      <th>song_id</th>
      <th>source_system_tab</th>
      <th>source_screen_name</th>
      <th>source_type</th>
      <th>target</th>
      <th>song_length</th>
      <th>genre_ids</th>
      <th>artist_name</th>
      <th>composer</th>
      <th>lyricist</th>
      <th>language</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>FGtllVqz18RPiwJj/edr2gV78zirAiY/9SmYvia+kCg=</td>
      <td>BBzumQNXUHKdEBOB7mAJuzok+IJA1c2Ryg/yzTF6tik=</td>
      <td>explore</td>
      <td>Explore</td>
      <td>online-playlist</td>
      <td>1</td>
      <td>206471.0</td>
      <td>359</td>
      <td>Bastille</td>
      <td>Dan Smith| Mark Crew</td>
      <td>NaN</td>
      <td>52.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Xumu+NIjS6QYVxDS4/t3SawvJ7viT9hPKXmf0RtLNx8=</td>
      <td>bhp/MpSNoqoxOIB+/l8WPqu6jldth4DIpCm3ayXnJqM=</td>
      <td>my library</td>
      <td>Local playlist more</td>
      <td>local-playlist</td>
      <td>1</td>
      <td>284584.0</td>
      <td>1259</td>
      <td>Various Artists</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>52.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Xumu+NIjS6QYVxDS4/t3SawvJ7viT9hPKXmf0RtLNx8=</td>
      <td>JNWfrrC7zNN7BdMpsISKa4Mw+xVJYNnxXh3/Epw7QgY=</td>
      <td>my library</td>
      <td>Local playlist more</td>
      <td>local-playlist</td>
      <td>1</td>
      <td>225396.0</td>
      <td>1259</td>
      <td>Nas</td>
      <td>N. Jones、W. Adams、J. Lordan、D. Ingle</td>
      <td>NaN</td>
      <td>52.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Xumu+NIjS6QYVxDS4/t3SawvJ7viT9hPKXmf0RtLNx8=</td>
      <td>2A87tzfnJTSWqD7gIZHisolhe4DMdzkbd6LzO1KHjNs=</td>
      <td>my library</td>
      <td>Local playlist more</td>
      <td>local-playlist</td>
      <td>1</td>
      <td>255512.0</td>
      <td>1019</td>
      <td>Soundway</td>
      <td>Kwadwo Donkoh</td>
      <td>NaN</td>
      <td>-1.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>FGtllVqz18RPiwJj/edr2gV78zirAiY/9SmYvia+kCg=</td>
      <td>3qm6XTZ6MOCU11x8FIVbAGH5l5uMkT3/ZalWG1oo2Gc=</td>
      <td>explore</td>
      <td>Explore</td>
      <td>online-playlist</td>
      <td>1</td>
      <td>187802.0</td>
      <td>1011</td>
      <td>Brett Young</td>
      <td>Brett Young| Kelly Archer| Justin Ebach</td>
      <td>NaN</td>
      <td>52.0</td>
    </tr>
  </tbody>
</table>
</div>



### Number of unique values in song_id, genre_id


```python
user_music_df['song_id'].nunique(), user_music_df['genre_ids'].nunique()
```




    (359966, 572)



### Count of not None values in each attribute


```python
user_music_df.count()
```




    msno                  7377418
    song_id               7377418
    source_system_tab     7352569
    source_screen_name    6962614
    source_type           7355879
    target                7377418
    song_length           7377304
    genre_ids             7258963
    artist_name           7377304
    composer              5701712
    lyricist              4198620
    language              7377268
    dtype: int64



### Visualize the counts of song genres


```python
# plot the top-20 frequent genre_ids 
df_genre = user_music_df.sample(n=5000)
df_genre = df_genre["genre_ids"].value_counts().sort_values(ascending=False)[:20]
df_genre = df_genre.sort_values(ascending=True)
ax  = df_genre.plot.barh(figsize=(15,8))
ax.set_ylabel("song genre ids")
ax.set_xlabel("Count")

```




    Text(0.5, 0, 'Count')




![png](kkboxmusicrecommendation_notebook_v7_files/kkboxmusicrecommendation_notebook_v7_51_1.png)


### Visualize the count of artists


```python

#selec the top-20 frequent artist_name
df_artist = user_music_df["artist_name"].value_counts().sort_values(ascending=False)[:20]
#plot in descending order in horizonal direction
df_artist = df_artist.sort_values(ascending=True)
ax  = df_artist.plot.barh(figsize=(15,10))
ax.set_ylabel("song artist_name")
ax.set_xlabel("Count")


# artist_name    
# composer       
# lyricist 
```




    Text(0.5, 0, 'Count')




![png](kkboxmusicrecommendation_notebook_v7_files/kkboxmusicrecommendation_notebook_v7_53_1.png)



```python
df_artist.head(10)
```




    The Chainsmokers     44215
    梁靜茹 (Fish Leong)     44290
    丁噹 (Della)           45762
    楊丞琳 (Rainie Yang)    46006
    蘇打綠 (Sodagreen)      47177
    蔡依林 (Jolin Tsai)     49055
    Eric 周興哲             49426
    A-Lin                52913
    Maroon 5             55151
    謝和弦 (R-chord)        57040
    Name: artist_name, dtype: int64



### Visualize the count of song composers


```python

fig, ax = plt.subplots(1, figsize=(15,8))
df_composer = user_music_df["composer"].value_counts().sort_values(ascending=False)[:20]
ax  = sns.barplot([i for i in df_composer.index],df_composer,ax= ax)
ax.set_xlabel("song composer")
ax.set_ylabel("Count")
```




    Text(0, 0.5, 'Count')




![png](kkboxmusicrecommendation_notebook_v7_files/kkboxmusicrecommendation_notebook_v7_56_1.png)



```python
df_composer.head(20).index
```




    Index(['周杰倫', '阿信', '林俊傑', '陳皓宇', 'JJ Lin', '張簡君偉', 'Eric Chou', '韋禮安',
           '八三夭 阿璞', 'R-chord', '怪獸', '吳青峰', '周湯豪', 'G.E.M. 鄧紫棋', '陳小霞', 'JerryC',
           '吳克群', '薛之謙', 'Rocoberry', '李榮浩'],
          dtype='object')



**Analysis:** we can see that features: counts of composer, artist, song genres are all long tail distribution. Most of users prefer the songs from particular composers, artists and song genres

### Analyse the relationship between target and song


```python
fig, ax = plt.subplots(3,1,figsize=(15,18))
# df = user_music_df[['source_system_tab','source_screen_name','source_type']]
# df['source_system_tab'].value_counts().plot.bar(rot=20,ax=ax[0])
# ax[0].set_xlabel("source_system_tab")
# ax[0].set_ylabel("count")

# df['source_screen_name'].value_counts().plot.bar(rot=30,ax=ax[1])
# ax[1].set_xlabel("source_screen_name")
# ax[1].set_ylabel("count")


# df['source_type'].value_counts().plot.bar(rot=20,ax=ax[2])
# ax[2].set_xlabel("source_type")
# ax[2].set_ylabel("count")

sns.countplot(y= 'source_system_tab',hue='target',
              order = user_music_df['source_system_tab'].value_counts().index,
              data=user_music_df,dodge=True, ax= ax[0])

sns.countplot(y= 'source_screen_name',hue='target',
              order = user_music_df['source_screen_name'].value_counts().index,
              data=user_music_df,dodge=True, ax= ax[1])


sns.countplot(y= 'source_type',hue='target',
              order = user_music_df['source_type'].value_counts().index,
              data=user_music_df,dodge=True, ax= ax[2])
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f269a90e128>




![png](kkboxmusicrecommendation_notebook_v7_files/kkboxmusicrecommendation_notebook_v7_60_1.png)


**Analysis:** We can see that local library and local playlist are the main sources that users repeat playing music and Most of users more prefer to play music from local library than to play music online

### Check the information of member data


```python
members_df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 34403 entries, 0 to 34402
    Data columns (total 7 columns):
     #   Column                  Non-Null Count  Dtype 
    ---  ------                  --------------  ----- 
     0   msno                    34403 non-null  object
     1   city                    34403 non-null  int64 
     2   bd                      34403 non-null  int64 
     3   gender                  14501 non-null  object
     4   registered_via          34403 non-null  int64 
     5   registration_init_time  34403 non-null  int64 
     6   expiration_date         34403 non-null  int64 
    dtypes: int64(5), object(2)
    memory usage: 1.8+ MB
    


```python
members_df["registration_init_time"] = pd.to_datetime(members_df["registration_init_time"], format="%Y%m%d")
members_df["expiration_date"] = pd.to_datetime(members_df["expiration_date"], format="%Y%m%d")
```

### Parse the datetime data


```python
members_df["registration_init_day"] = members_df["registration_init_time"].dt.day
members_df["registration_init_month"] = members_df["registration_init_time"].dt.month
members_df["registration_init_year"] = members_df["registration_init_time"].dt.year
members_df["expiration_day"] = members_df["expiration_date"].dt.day
members_df["expiration_month"] = members_df["expiration_date"].dt.month
members_df["expiration_year"] = members_df["expiration_date"].dt.year
members_df = members_df.drop(columns = ["registration_init_time", "expiration_date"],axis=1)
members_df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 34403 entries, 0 to 34402
    Data columns (total 11 columns):
     #   Column                   Non-Null Count  Dtype 
    ---  ------                   --------------  ----- 
     0   msno                     34403 non-null  object
     1   city                     34403 non-null  int64 
     2   bd                       34403 non-null  int64 
     3   gender                   14501 non-null  object
     4   registered_via           34403 non-null  int64 
     5   registration_init_day    34403 non-null  int64 
     6   registration_init_month  34403 non-null  int64 
     7   registration_init_year   34403 non-null  int64 
     8   expiration_day           34403 non-null  int64 
     9   expiration_month         34403 non-null  int64 
     10  expiration_year          34403 non-null  int64 
    dtypes: int64(9), object(2)
    memory usage: 2.9+ MB
    

### Merge member data with training data


```python
member_music_df = user_music_df.merge(members_df,on='msno',how="left", copy=False)

#after merging, the axis used to merge becomes object type,so need to convert it back to category type
member_music_df["msno"] = member_music_df["msno"].astype("category")
member_music_df["song_id"] = member_music_df["song_id"].astype("category")
member_music_df.head()

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>msno</th>
      <th>song_id</th>
      <th>source_system_tab</th>
      <th>source_screen_name</th>
      <th>source_type</th>
      <th>target</th>
      <th>song_length</th>
      <th>genre_ids</th>
      <th>artist_name</th>
      <th>composer</th>
      <th>lyricist</th>
      <th>language</th>
      <th>city</th>
      <th>bd</th>
      <th>gender</th>
      <th>registered_via</th>
      <th>registration_init_day</th>
      <th>registration_init_month</th>
      <th>registration_init_year</th>
      <th>expiration_day</th>
      <th>expiration_month</th>
      <th>expiration_year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>FGtllVqz18RPiwJj/edr2gV78zirAiY/9SmYvia+kCg=</td>
      <td>BBzumQNXUHKdEBOB7mAJuzok+IJA1c2Ryg/yzTF6tik=</td>
      <td>explore</td>
      <td>Explore</td>
      <td>online-playlist</td>
      <td>1</td>
      <td>206471.0</td>
      <td>359</td>
      <td>Bastille</td>
      <td>Dan Smith| Mark Crew</td>
      <td>NaN</td>
      <td>52.0</td>
      <td>1</td>
      <td>0</td>
      <td>NaN</td>
      <td>7</td>
      <td>2</td>
      <td>1</td>
      <td>2012</td>
      <td>5</td>
      <td>10</td>
      <td>2017</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Xumu+NIjS6QYVxDS4/t3SawvJ7viT9hPKXmf0RtLNx8=</td>
      <td>bhp/MpSNoqoxOIB+/l8WPqu6jldth4DIpCm3ayXnJqM=</td>
      <td>my library</td>
      <td>Local playlist more</td>
      <td>local-playlist</td>
      <td>1</td>
      <td>284584.0</td>
      <td>1259</td>
      <td>Various Artists</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>52.0</td>
      <td>13</td>
      <td>24</td>
      <td>female</td>
      <td>9</td>
      <td>25</td>
      <td>5</td>
      <td>2011</td>
      <td>11</td>
      <td>9</td>
      <td>2017</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Xumu+NIjS6QYVxDS4/t3SawvJ7viT9hPKXmf0RtLNx8=</td>
      <td>JNWfrrC7zNN7BdMpsISKa4Mw+xVJYNnxXh3/Epw7QgY=</td>
      <td>my library</td>
      <td>Local playlist more</td>
      <td>local-playlist</td>
      <td>1</td>
      <td>225396.0</td>
      <td>1259</td>
      <td>Nas</td>
      <td>N. Jones、W. Adams、J. Lordan、D. Ingle</td>
      <td>NaN</td>
      <td>52.0</td>
      <td>13</td>
      <td>24</td>
      <td>female</td>
      <td>9</td>
      <td>25</td>
      <td>5</td>
      <td>2011</td>
      <td>11</td>
      <td>9</td>
      <td>2017</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Xumu+NIjS6QYVxDS4/t3SawvJ7viT9hPKXmf0RtLNx8=</td>
      <td>2A87tzfnJTSWqD7gIZHisolhe4DMdzkbd6LzO1KHjNs=</td>
      <td>my library</td>
      <td>Local playlist more</td>
      <td>local-playlist</td>
      <td>1</td>
      <td>255512.0</td>
      <td>1019</td>
      <td>Soundway</td>
      <td>Kwadwo Donkoh</td>
      <td>NaN</td>
      <td>-1.0</td>
      <td>13</td>
      <td>24</td>
      <td>female</td>
      <td>9</td>
      <td>25</td>
      <td>5</td>
      <td>2011</td>
      <td>11</td>
      <td>9</td>
      <td>2017</td>
    </tr>
    <tr>
      <th>4</th>
      <td>FGtllVqz18RPiwJj/edr2gV78zirAiY/9SmYvia+kCg=</td>
      <td>3qm6XTZ6MOCU11x8FIVbAGH5l5uMkT3/ZalWG1oo2Gc=</td>
      <td>explore</td>
      <td>Explore</td>
      <td>online-playlist</td>
      <td>1</td>
      <td>187802.0</td>
      <td>1011</td>
      <td>Brett Young</td>
      <td>Brett Young| Kelly Archer| Justin Ebach</td>
      <td>NaN</td>
      <td>52.0</td>
      <td>1</td>
      <td>0</td>
      <td>NaN</td>
      <td>7</td>
      <td>2</td>
      <td>1</td>
      <td>2012</td>
      <td>5</td>
      <td>10</td>
      <td>2017</td>
    </tr>
  </tbody>
</table>
</div>




```python
member_music_df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 7377418 entries, 0 to 7377417
    Data columns (total 22 columns):
     #   Column                   Dtype   
    ---  ------                   -----   
     0   msno                     category
     1   song_id                  category
     2   source_system_tab        object  
     3   source_screen_name       object  
     4   source_type              object  
     5   target                   int64   
     6   song_length              float64 
     7   genre_ids                object  
     8   artist_name              object  
     9   composer                 object  
     10  lyricist                 object  
     11  language                 float64 
     12  city                     int64   
     13  bd                       int64   
     14  gender                   object  
     15  registered_via           int64   
     16  registration_init_day    int64   
     17  registration_init_month  int64   
     18  registration_init_year   int64   
     19  expiration_day           int64   
     20  expiration_month         int64   
     21  expiration_year          int64   
    dtypes: category(2), float64(2), int64(10), object(8)
    memory usage: 1.2+ GB
    


```python
member_music_df.count()
```




    msno                       7377418
    song_id                    7377418
    source_system_tab          7352569
    source_screen_name         6962614
    source_type                7355879
    target                     7377418
    song_length                7377304
    genre_ids                  7258963
    artist_name                7377304
    composer                   5701712
    lyricist                   4198620
    language                   7377268
    city                       7377418
    bd                         7377418
    gender                     4415939
    registered_via             7377418
    registration_init_day      7377418
    registration_init_month    7377418
    registration_init_year     7377418
    expiration_day             7377418
    expiration_month           7377418
    expiration_year            7377418
    dtype: int64




```python
member_music_df['bd'].describe()
```




    count    7.377418e+06
    mean     1.753927e+01
    std      2.155447e+01
    min     -4.300000e+01
    25%      0.000000e+00
    50%      2.100000e+01
    75%      2.900000e+01
    max      1.051000e+03
    Name: bd, dtype: float64



### Visualize distribution of age: bd attribution
Note: Since this attribute has outliers, I use remove the data that lies outside range [0,100] by replacing them with NaN


```python
fig, ax = plt.subplots(2, figsize= (15,8))
age_df = member_music_df['bd'].loc[(member_music_df['bd']>0) & (member_music_df['bd']<100)]
age_df.hist(ax = ax[0])

ax[0].set_ylabel("count")

member_music_df['bd'].loc[(member_music_df['bd']<0) | (member_music_df['bd']>100)].hist(ax = ax[1])
ax[1].set_xlabel("age")
ax[1].set_ylabel("count")
```




    Text(0, 0.5, 'count')




![png](kkboxmusicrecommendation_notebook_v7_files/kkboxmusicrecommendation_notebook_v7_73_1.png)


**Analysis:** We can see that bd/age column has outliers outside range [0,100], so we want to replace the incorrect bd with NaN


```python
member_music_df['bd'].loc[(member_music_df['bd']<=0) | (member_music_df['bd']>=100)]= np.nan
```

### Check the correlation between attributes using correlation matrix and visualize them wiht heatmap


```python
member_music_df.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>target</th>
      <th>song_length</th>
      <th>language</th>
      <th>city</th>
      <th>bd</th>
      <th>registered_via</th>
      <th>registration_init_day</th>
      <th>registration_init_month</th>
      <th>registration_init_year</th>
      <th>expiration_day</th>
      <th>expiration_month</th>
      <th>expiration_year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>7.377418e+06</td>
      <td>7.377304e+06</td>
      <td>7.377268e+06</td>
      <td>7.377418e+06</td>
      <td>4.430216e+06</td>
      <td>7.377418e+06</td>
      <td>7.377418e+06</td>
      <td>7.377418e+06</td>
      <td>7.377418e+06</td>
      <td>7.377418e+06</td>
      <td>7.377418e+06</td>
      <td>7.377418e+06</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>5.035171e-01</td>
      <td>2.451210e+05</td>
      <td>1.860933e+01</td>
      <td>7.511399e+00</td>
      <td>2.872200e+01</td>
      <td>6.794068e+00</td>
      <td>1.581532e+01</td>
      <td>6.832306e+00</td>
      <td>2.012741e+03</td>
      <td>1.562338e+01</td>
      <td>8.341742e+00</td>
      <td>2.017072e+03</td>
    </tr>
    <tr>
      <th>std</th>
      <td>4.999877e-01</td>
      <td>6.734471e+04</td>
      <td>2.117681e+01</td>
      <td>6.641625e+00</td>
      <td>8.634326e+00</td>
      <td>2.275774e+00</td>
      <td>8.768549e+00</td>
      <td>3.700723e+00</td>
      <td>3.018861e+00</td>
      <td>9.107235e+00</td>
      <td>2.511360e+00</td>
      <td>3.982536e-01</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000e+00</td>
      <td>1.393000e+03</td>
      <td>-1.000000e+00</td>
      <td>1.000000e+00</td>
      <td>2.000000e+00</td>
      <td>3.000000e+00</td>
      <td>1.000000e+00</td>
      <td>1.000000e+00</td>
      <td>2.004000e+03</td>
      <td>1.000000e+00</td>
      <td>1.000000e+00</td>
      <td>1.970000e+03</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.000000e+00</td>
      <td>2.147260e+05</td>
      <td>3.000000e+00</td>
      <td>1.000000e+00</td>
      <td>2.300000e+01</td>
      <td>4.000000e+00</td>
      <td>8.000000e+00</td>
      <td>3.000000e+00</td>
      <td>2.011000e+03</td>
      <td>8.000000e+00</td>
      <td>9.000000e+00</td>
      <td>2.017000e+03</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>1.000000e+00</td>
      <td>2.418120e+05</td>
      <td>3.000000e+00</td>
      <td>5.000000e+00</td>
      <td>2.700000e+01</td>
      <td>7.000000e+00</td>
      <td>1.600000e+01</td>
      <td>7.000000e+00</td>
      <td>2.013000e+03</td>
      <td>1.500000e+01</td>
      <td>9.000000e+00</td>
      <td>2.017000e+03</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>1.000000e+00</td>
      <td>2.721600e+05</td>
      <td>5.200000e+01</td>
      <td>1.300000e+01</td>
      <td>3.300000e+01</td>
      <td>9.000000e+00</td>
      <td>2.300000e+01</td>
      <td>1.000000e+01</td>
      <td>2.015000e+03</td>
      <td>2.300000e+01</td>
      <td>1.000000e+01</td>
      <td>2.017000e+03</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.000000e+00</td>
      <td>1.085171e+07</td>
      <td>5.900000e+01</td>
      <td>2.200000e+01</td>
      <td>9.500000e+01</td>
      <td>1.300000e+01</td>
      <td>3.100000e+01</td>
      <td>1.200000e+01</td>
      <td>2.017000e+03</td>
      <td>3.100000e+01</td>
      <td>1.200000e+01</td>
      <td>2.020000e+03</td>
    </tr>
  </tbody>
</table>
</div>




```python
#dataframe.corr() is used to find the pairwise correlation of all columns in the dataframe. 
#Any na values are automatically excluded. For any non-numeric data type columns in the data frame it is ignored.
corr_matrix = member_music_df.corr()
_ = sns.heatmap(corr_matrix)

```


![png](kkboxmusicrecommendation_notebook_v7_files/kkboxmusicrecommendation_notebook_v7_78_0.png)


**Analysis:** From the correlation matrix above, it is hard for us to see which attributes are most related to target as the correlation values of target are very similar.In this case, we choose to use all features to train models later

It seems like registration_init_year is negatively correlative to bd/age, city, registered_via. Song_length is also negatively correlative to language.



```python
#print the top threee attributes that have the strongest correlation with "Target" and the corresponding correlation coefficients.
corr = corr_matrix['target'].sort_values(ascending= False)
for x in corr.index[1:4].to_list():
    print("{} {}".format(x, corr[x]))
print("")
#print the top threee attributes that have the strongest correlation with "song_length" and the corresponding correlation coefficients.
corr = corr_matrix['song_length'].sort_values(ascending= False)
for x in corr.index[1:4].to_list():
    print("{} {}".format(x, corr[x]))
print("")
#print the top threee attributes that have the strongest correlation with "language" and the corresponding correlation coefficients.
corr = corr_matrix['language'].sort_values(ascending= False)
for x in corr.index[1:4].to_list():
    print("{} {}".format(x, corr[x]))
print("")
#print the top threee attributes that have the strongest correlation with "city" and the corresponding correlation coefficients.
corr = corr_matrix['city'].sort_values(ascending= False)
for x in corr.index[1:4].to_list():
    print("{} {}".format(x, corr[x]))
print("")
#print the top threee attributes that have the strongest correlation with "bd" and the corresponding correlation coefficients.
corr = corr_matrix['bd'].sort_values(ascending= False)
for x in corr.index[1:4].to_list():
    print("{} {}".format(x, corr[x]))
print("")
#print the top threee attributes that have the strongest correlation with "registered_via" and the corresponding correlation coefficients.
corr = corr_matrix['registered_via'].sort_values(ascending= False)
for x in corr.index[1:4].to_list():
    print("{} {}".format(x, corr[x]))
print("")
corr = corr_matrix['registration_init_day'].sort_values(ascending= False)
for x in corr.index[1:4].to_list():
    print("{} {}".format(x, corr[x]))
print("")
corr = corr_matrix['registration_init_month'].sort_values(ascending= False)
for x in corr.index[1:4].to_list():
    print("{} {}".format(x, corr[x]))
print("")
corr = corr_matrix['registration_init_year'].sort_values(ascending= False)
for x in corr.index[1:4].to_list():
    print("{} {}".format(x, corr[x]))
print("")
corr = corr_matrix['expiration_day'].sort_values(ascending= False)
for x in corr.index[1:4].to_list():
    print("{} {}".format(x, corr[x]))
print("")
corr = corr_matrix['expiration_month'].sort_values(ascending= False)
for x in corr.index[1:4].to_list():
    print("{} {}".format(x, corr[x]))
print("")
corr = corr_matrix['expiration_year'].sort_values(ascending= False)
for x in corr.index[1:4].to_list():
    print("{} {}".format(x, corr[x]))
print("")


```

    expiration_year 0.042248332355979766
    city 0.01211438566189457
    expiration_month 0.011817072086387569
    
    bd 0.009861302779254176
    city 0.005184912771179072
    expiration_year 0.00457185870016758
    
    registration_init_year 0.009070490482763404
    registration_init_day 0.001510510575428178
    bd 0.001107978394135987
    
    expiration_year 0.15014690465127595
    registered_via 0.0737556175747622
    target 0.01211438566189457
    
    registered_via 0.1753390015877422
    expiration_day 0.056335854806629254
    expiration_month 0.032935904360496926
    
    bd 0.1753390015877422
    expiration_year 0.08413460079453493
    city 0.0737556175747622
    
    expiration_day 0.1493505099924221
    registration_init_month 0.04443692475737983
    registered_via 0.02554331305533987
    
    expiration_month 0.056911114419175665
    registration_init_day 0.04443692475737983
    bd 0.005399463812914416
    
    language 0.009070490482763404
    target -0.00196242388069252
    song_length -0.007434856516605977
    
    registration_init_day 0.1493505099924221
    registered_via 0.05695618668075027
    bd 0.056335854806629254
    
    registered_via 0.0647318000666518
    registration_init_month 0.056911114419175665
    bd 0.032935904360496926
    
    city 0.15014690465127595
    registered_via 0.08413460079453493
    target 0.042248332355979766
    
    

## Use Bivariate Plot to visualize correlation between pair of attributes

Visualize Relationship between bd/age and city


```python
#graph
fig, ax = plt.subplots(1,1,figsize=(10,8), sharex=False)
plt.scatter(x = member_music_df['city'],y = member_music_df['bd'])
ax.set_ylabel("bd")
ax.set_xlabel("city")
plt.show()


```


![png](kkboxmusicrecommendation_notebook_v7_files/kkboxmusicrecommendation_notebook_v7_83_0.png)


Visualize Relationship between target and expiration year using bivariate plot to see if expiration date matters



```python
#graph
fig, ax = plt.subplots(1,1,figsize=(10,8), sharex=False)
plt.scatter(x = member_music_df['target'],y = member_music_df['expiration_year'])
ax.set_ylabel("expiration_year")
ax.set_xlabel("target")
plt.show()
```


![png](kkboxmusicrecommendation_notebook_v7_files/kkboxmusicrecommendation_notebook_v7_85_0.png)


**Analysis:** it seems like there is no correlation between expiration year and target.


```python
#user的musiclist里面可能重听的music 
print(train_df.target.value_counts()*100/train_df.target.value_counts().sum())
print('unique songs ',len(train_df.song_id.unique()))
#unique() = Return unique values of Series object.
#len() to find unqiue sound.
```

    1    50.351708
    0    49.648292
    Name: target, dtype: float64
    unique songs  359966
    

Count of top 50 songs which users repeat listening to 


```python
repeats=train_df[train_df.target==1]
song_repeats=repeats.groupby('song_id',as_index=False).msno.count()
song_repeats.columns=['song_id','count']
##merge together 2 dataframe and create a new dataframe
song_repeats=pd.DataFrame(song_repeats).merge(song_df,left_on='song_id',right_on='song_id')
print("Print top 50 songs repeated")
repeats.song_id.value_counts().head(50)
```

    Print top 50 songs repeated
    




    reXuGcEWDDCnL0K3Th//3DFG4S1ACSpJMzA+CFipo1g=    10885
    T86YHdD4C9JSc274b1IlMkLuNdz4BQRB50fWWE7hx9g=    10556
    FynUyq0+drmIARmK1JZ/qcjNZ7DKkqTY6/0O0lTzNUI=     9808
    wBTWuHbjdjxnG1lQcbqnK4FddV24rUhuyrYLd9c/hmk=     9411
    PgRtmmESVNtWjoZHO5a1r21vIz9sVZmcJJpFCbRa1LI=     9004
    U9kojfZSKaiWOW94PKh1Riyv/zUWxmBRmv0XInQWLGw=     8787
    YN4T/yvvXtYrBVN8KTnieiQohHL3T9fnzUkbLWcgLro=     8780
    M9rAajz4dYuRhZ7jLvf9RRayVA3os61X/XXHEuW4giA=     8403
    43Qm2YzsP99P5wm37B1JIhezUcQ/1CDjYlQx6rBbz2U=     8112
    J4qKkLIoW7aYACuTupHLAPZYmRp08en1AEux+GSUzdw=     7903
    cy10N2j2sdY/X4BDUcMu2Iumfz7pV3tqE5iEaup2yGI=     7725
    750RprmFfLV0bymtDH88g24pLZGVi5VpBAI300P6UOA=     7608
    IKMFuL0f5Y8c63Hg9BXkeNJjE0z8yf3gMt/tOxF4QNE=     7224
    +SstqMwhQPBQFTPBhLKPT642IiBDXzZFwlzsLl4cGXo=     7061
    DLBDZhOoW7zd7GBV99bi92ZXYUS26lzV+jJKbHshP5c=     6901
    v/3onppBGoSpGsWb8iaCIO8eX5+iacbH5a4ZUhT7N54=     6879
    p/yR06j/RQ2J6yGCFL0K+1R06OeG+eXcwxRgOHDo/Tk=     6536
    Xpjwi8UAE2Vv9PZ6cZnhc58MCtl3cKZEO1sdAkqJ4mo=     6399
    OaEbZ6TJ1NePtNUeEgWsvFLeopkSln9WQu8PBR5B3+A=     6187
    BITuBuNyXQydJcjDL2BUnCu4/IXaJg5IPOuycc/4dtY=     6160
    BgqjNqzsyCpEGvxyUmktvHC8WO5+FQO/pQTaZ4broMU=     6140
    3VkD5ekIf5duJm1hmYTZlXjyl0zqV8wCzuAh3uocfCg=     6012
    8Ckw1wek5d6oEsNUoM4P5iag86TaEmyLwdtrckL0Re8=     6003
    n+pMhj/jpCnpiUcSDl4k3i9FJODDddEXmpE48/HczTI=     5787
    WL4ipO3Mx9pxd4FMs69ha6o9541+fLeOow67Qkrfnro=     5784
    /70HjygVDhHsKBoV8mmsBg/WduSgs4+Zg6GfzhUQbdk=     5588
    L6w2d0w84FjTvFr+BhMfgu7dZAsGiOqUGmvvxIG3gvQ=     5480
    fEAIgFRWmhXmo6m3ukQeqRksZCcO/7CjkqNckRHiVQo=     5460
    +Sm75wnBf/sjm/QMUAFx8N+Ae04kWCXGlgH50tTeM6c=     5412
    VkDBgh89umc9m6uAEfD6LXngetyGhln4vh/ArCGO0nY=     5361
    fCCmIa0Y5m+MCGbQga31MOLTIqi7ddgXvkjFPmfslGw=     5305
    +LztcJcPEEwsikk6+K5udm06XJQMzR4+lzavKLUyE0k=     5298
    o9HWMBZMeIPnYEpSuscGoORKE44sj3BYOdvGuIi0P68=     5233
    QZBm8SOwnEjNfCpgsKBBGPMGET6y6XaQgnJiirspW7I=     5224
    ClazTFnk6r0Bnuie44bocdNMM3rdlrq0bCGAsGUWcHE=     5202
    wp1gSQ4LlMEF6bzvEaJl8VdHlAj/EJMTJ0ASrXeddbo=     5110
    THqGcrzQyUhBn1NI/+Iptc1vKtxBIEg0uA8iaoJnO1Q=     5086
    ys+EL8Sok4HC4i7sDY0+slDNGVZ8+uOQi6TQ6g8VSF4=     5012
    zHqZ07gn+YvF36FWzv9+y8KiCMhYhdAUS+vSIKY3UZY=     5001
    8f/T4ohROj1wa25YHMItOW2/wJhRXZM0+T5/2p86COc=     4982
    G/4+VCRLpfjQJ4SAwMDcf+W8PTw0eOBRgFvg4fHUOO8=     4956
    KZ5hwP74wRO6kRapVIprwodtNdVD2EVD3hkZmmyXFPk=     4888
    MtFK4NN8Kv1k/xPA3wb8SQaP/jWee52FAaC1s9NFsU4=     4813
    UQeOwfhcqgEcIwp3cgNiLGW1237Qjpvqzt/asQimVp0=     4778
    JA6C0GEK1sSCVbHyqtruH/ARD1NKolYrw7HXy6EVNAc=     4766
    8qWeDv6RTv+hYJxW94e7n6HBzHPGPEZW9FuGhj6pPhQ=     4761
    35dx60z4m4+Lg+qIS0l2A8vspbthqnpTylWUu51jW+4=     4679
    r4lUPUkz3tAgIWaEyrSYVCxX1yz8PnlVuQz+To0Pd+c=     4650
    1PR/lVwL4VeYcZjexwBJ2NOSTfgh8JoVxWCunnbJO/8=     4592
    7EnDBkQYJpipCyRd9JBsug4iKnfAunUXc14/96cNotg=     4571
    Name: song_id, dtype: int64



# 2. Data Preprocessing


Note: This section is to show how to preprocess data. We can also directly start from Step 3 for data extract, transformation and load using integrated transformation function and skip this step if necessary

## 2.1 Filling missing values

### List of features that contain missing values


```python
missing_value_cols = [c for c in member_music_df.columns if member_music_df[c].isnull().any()]
missing_value_cols
```




    ['source_system_tab',
     'source_screen_name',
     'source_type',
     'song_length',
     'genre_ids',
     'artist_name',
     'composer',
     'lyricist',
     'language',
     'bd',
     'gender']




```python
member_music_df.count()
```




    msno                       7377418
    song_id                    7377418
    source_system_tab          7352569
    source_screen_name         6962614
    source_type                7355879
    target                     7377418
    song_length                7377304
    genre_ids                  7258963
    artist_name                7377304
    composer                   5701712
    lyricist                   4198620
    language                   7377268
    city                       7377418
    bd                         4430216
    gender                     4415939
    registered_via             7377418
    registration_init_day      7377418
    registration_init_month    7377418
    registration_init_year     7377418
    expiration_day             7377418
    expiration_month           7377418
    expiration_year            7377418
    dtype: int64




```python
# list of columns with missing values
# ['source_system_tab',
#  'source_screen_name',
#  'source_type',
#  'song_length',
#  'genre_ids',
#  'artist_name',
#  'composer',
#  'lyricist',
#  'language',
#  'bd',
#  'gender']

def fill_missing_value_v1(x):
    # fill missing values with the most frequent values
    return x.fillna(x.value_counts().sort_values(ascending=False).index[0])
    
    

categorical_ls = ['source_system_tab', 'source_screen_name','source_type','genre_ids','artist_name','composer',
 'lyricist','gender']


numerical_ls = ['song_length','language','bd']
# Fill missing values 
for index in numerical_ls:
    member_music_df[index].fillna(member_music_df[index].median(), inplace=True)
for index in categorical_ls:
    member_music_df[index].fillna("no_data", inplace=True)

```

### Count of features after filling missing values


```python
member_music_df.count()
```




    msno                       7377418
    song_id                    7377418
    source_system_tab          7377418
    source_screen_name         7377418
    source_type                7377418
    target                     7377418
    song_length                7377418
    genre_ids                  7377418
    artist_name                7377418
    composer                   7377418
    lyricist                   7377418
    language                   7377418
    city                       7377418
    bd                         7377418
    gender                     7377418
    registered_via             7377418
    registration_init_day      7377418
    registration_init_month    7377418
    registration_init_year     7377418
    expiration_day             7377418
    expiration_month           7377418
    expiration_year            7377418
    dtype: int64




```python

member_music_df[numerical_ls].head(100)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>song_length</th>
      <th>language</th>
      <th>bd</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>206471.0</td>
      <td>52.0</td>
      <td>27.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>284584.0</td>
      <td>52.0</td>
      <td>24.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>225396.0</td>
      <td>52.0</td>
      <td>24.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>255512.0</td>
      <td>-1.0</td>
      <td>24.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>187802.0</td>
      <td>52.0</td>
      <td>27.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>95</th>
      <td>333024.0</td>
      <td>3.0</td>
      <td>27.0</td>
    </tr>
    <tr>
      <th>96</th>
      <td>288391.0</td>
      <td>3.0</td>
      <td>46.0</td>
    </tr>
    <tr>
      <th>97</th>
      <td>279196.0</td>
      <td>3.0</td>
      <td>46.0</td>
    </tr>
    <tr>
      <th>98</th>
      <td>240744.0</td>
      <td>3.0</td>
      <td>46.0</td>
    </tr>
    <tr>
      <th>99</th>
      <td>221622.0</td>
      <td>3.0</td>
      <td>46.0</td>
    </tr>
  </tbody>
</table>
<p>100 rows × 3 columns</p>
</div>




```python

member_music_df[categorical_ls].head(100)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>source_system_tab</th>
      <th>source_screen_name</th>
      <th>source_type</th>
      <th>genre_ids</th>
      <th>artist_name</th>
      <th>composer</th>
      <th>lyricist</th>
      <th>gender</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>explore</td>
      <td>Explore</td>
      <td>online-playlist</td>
      <td>359</td>
      <td>Bastille</td>
      <td>Dan Smith| Mark Crew</td>
      <td>no_data</td>
      <td>no_data</td>
    </tr>
    <tr>
      <th>1</th>
      <td>my library</td>
      <td>Local playlist more</td>
      <td>local-playlist</td>
      <td>1259</td>
      <td>Various Artists</td>
      <td>no_data</td>
      <td>no_data</td>
      <td>female</td>
    </tr>
    <tr>
      <th>2</th>
      <td>my library</td>
      <td>Local playlist more</td>
      <td>local-playlist</td>
      <td>1259</td>
      <td>Nas</td>
      <td>N. Jones、W. Adams、J. Lordan、D. Ingle</td>
      <td>no_data</td>
      <td>female</td>
    </tr>
    <tr>
      <th>3</th>
      <td>my library</td>
      <td>Local playlist more</td>
      <td>local-playlist</td>
      <td>1019</td>
      <td>Soundway</td>
      <td>Kwadwo Donkoh</td>
      <td>no_data</td>
      <td>female</td>
    </tr>
    <tr>
      <th>4</th>
      <td>explore</td>
      <td>Explore</td>
      <td>online-playlist</td>
      <td>1011</td>
      <td>Brett Young</td>
      <td>Brett Young| Kelly Archer| Justin Ebach</td>
      <td>no_data</td>
      <td>no_data</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>95</th>
      <td>my library</td>
      <td>no_data</td>
      <td>local-library</td>
      <td>458</td>
      <td>楊乃文 (Naiwen Yang)</td>
      <td>黃建為</td>
      <td>葛大為</td>
      <td>male</td>
    </tr>
    <tr>
      <th>96</th>
      <td>my library</td>
      <td>Local playlist more</td>
      <td>local-library</td>
      <td>458</td>
      <td>陳奕迅 (Eason Chan)</td>
      <td>Jun Jie Lin</td>
      <td>no_data</td>
      <td>female</td>
    </tr>
    <tr>
      <th>97</th>
      <td>my library</td>
      <td>Local playlist more</td>
      <td>local-library</td>
      <td>458</td>
      <td>周杰倫 (Jay Chou)</td>
      <td>周杰倫</td>
      <td>方文山</td>
      <td>female</td>
    </tr>
    <tr>
      <th>98</th>
      <td>my library</td>
      <td>Local playlist more</td>
      <td>local-library</td>
      <td>465</td>
      <td>范瑋琪 (Christine Fan)</td>
      <td>非非</td>
      <td>非非</td>
      <td>female</td>
    </tr>
    <tr>
      <th>99</th>
      <td>my library</td>
      <td>Local playlist more</td>
      <td>local-library</td>
      <td>465|1259</td>
      <td>玖壹壹</td>
      <td>陳皓宇</td>
      <td>廖建至|洪瑜鴻</td>
      <td>female</td>
    </tr>
  </tbody>
</table>
<p>100 rows × 8 columns</p>
</div>



## 2.2 Data Transformation
We can see that the columns like genre_ids, composer, lyricist have multiple values in a cell. In this case, the count of genres, composers, lyricist could be useful information as well


```python
member_music_df.columns
```




    Index(['msno', 'song_id', 'source_system_tab', 'source_screen_name',
           'source_type', 'target', 'song_length', 'genre_ids', 'artist_name',
           'composer', 'lyricist', 'language', 'city', 'bd', 'gender',
           'registered_via', 'registration_init_day', 'registration_init_month',
           'registration_init_year', 'expiration_day', 'expiration_month',
           'expiration_year'],
          dtype='object')




```python
member_music_df.genre_ids.nunique(), member_music_df.composer.nunique(), member_music_df.lyricist.nunique()
```




    (573, 76065, 33889)




```python

def count_items(x):
    if x =="no_data":
        return 0
    return sum(map(x.count, ['|', '/', '\\', ';',','])) + 1

member_music_df['genre_count']=  member_music_df['genre_ids'].apply(count_items)
member_music_df['composer_count']=  member_music_df['composer'].apply(count_items)
member_music_df['lyricist_count']=  member_music_df['lyricist'].apply(count_items)
```


```python
member_music_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>msno</th>
      <th>song_id</th>
      <th>source_system_tab</th>
      <th>source_screen_name</th>
      <th>source_type</th>
      <th>target</th>
      <th>song_length</th>
      <th>genre_ids</th>
      <th>artist_name</th>
      <th>composer</th>
      <th>lyricist</th>
      <th>language</th>
      <th>city</th>
      <th>bd</th>
      <th>gender</th>
      <th>registered_via</th>
      <th>registration_init_day</th>
      <th>registration_init_month</th>
      <th>registration_init_year</th>
      <th>expiration_day</th>
      <th>expiration_month</th>
      <th>expiration_year</th>
      <th>genre_count</th>
      <th>composer_count</th>
      <th>lyricist_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>FGtllVqz18RPiwJj/edr2gV78zirAiY/9SmYvia+kCg=</td>
      <td>BBzumQNXUHKdEBOB7mAJuzok+IJA1c2Ryg/yzTF6tik=</td>
      <td>explore</td>
      <td>Explore</td>
      <td>online-playlist</td>
      <td>1</td>
      <td>206471.0</td>
      <td>359</td>
      <td>Bastille</td>
      <td>Dan Smith| Mark Crew</td>
      <td>no_data</td>
      <td>52.0</td>
      <td>1</td>
      <td>27.0</td>
      <td>no_data</td>
      <td>7</td>
      <td>2</td>
      <td>1</td>
      <td>2012</td>
      <td>5</td>
      <td>10</td>
      <td>2017</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Xumu+NIjS6QYVxDS4/t3SawvJ7viT9hPKXmf0RtLNx8=</td>
      <td>bhp/MpSNoqoxOIB+/l8WPqu6jldth4DIpCm3ayXnJqM=</td>
      <td>my library</td>
      <td>Local playlist more</td>
      <td>local-playlist</td>
      <td>1</td>
      <td>284584.0</td>
      <td>1259</td>
      <td>Various Artists</td>
      <td>no_data</td>
      <td>no_data</td>
      <td>52.0</td>
      <td>13</td>
      <td>24.0</td>
      <td>female</td>
      <td>9</td>
      <td>25</td>
      <td>5</td>
      <td>2011</td>
      <td>11</td>
      <td>9</td>
      <td>2017</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Xumu+NIjS6QYVxDS4/t3SawvJ7viT9hPKXmf0RtLNx8=</td>
      <td>JNWfrrC7zNN7BdMpsISKa4Mw+xVJYNnxXh3/Epw7QgY=</td>
      <td>my library</td>
      <td>Local playlist more</td>
      <td>local-playlist</td>
      <td>1</td>
      <td>225396.0</td>
      <td>1259</td>
      <td>Nas</td>
      <td>N. Jones、W. Adams、J. Lordan、D. Ingle</td>
      <td>no_data</td>
      <td>52.0</td>
      <td>13</td>
      <td>24.0</td>
      <td>female</td>
      <td>9</td>
      <td>25</td>
      <td>5</td>
      <td>2011</td>
      <td>11</td>
      <td>9</td>
      <td>2017</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Xumu+NIjS6QYVxDS4/t3SawvJ7viT9hPKXmf0RtLNx8=</td>
      <td>2A87tzfnJTSWqD7gIZHisolhe4DMdzkbd6LzO1KHjNs=</td>
      <td>my library</td>
      <td>Local playlist more</td>
      <td>local-playlist</td>
      <td>1</td>
      <td>255512.0</td>
      <td>1019</td>
      <td>Soundway</td>
      <td>Kwadwo Donkoh</td>
      <td>no_data</td>
      <td>-1.0</td>
      <td>13</td>
      <td>24.0</td>
      <td>female</td>
      <td>9</td>
      <td>25</td>
      <td>5</td>
      <td>2011</td>
      <td>11</td>
      <td>9</td>
      <td>2017</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>FGtllVqz18RPiwJj/edr2gV78zirAiY/9SmYvia+kCg=</td>
      <td>3qm6XTZ6MOCU11x8FIVbAGH5l5uMkT3/ZalWG1oo2Gc=</td>
      <td>explore</td>
      <td>Explore</td>
      <td>online-playlist</td>
      <td>1</td>
      <td>187802.0</td>
      <td>1011</td>
      <td>Brett Young</td>
      <td>Brett Young| Kelly Archer| Justin Ebach</td>
      <td>no_data</td>
      <td>52.0</td>
      <td>1</td>
      <td>27.0</td>
      <td>no_data</td>
      <td>7</td>
      <td>2</td>
      <td>1</td>
      <td>2012</td>
      <td>5</td>
      <td>10</td>
      <td>2017</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
member_music_df.info()
```

# 3. Data Extract, Transform and Load (ETL)

This step build a data pipeline to clean and transform data directly right before training model. We can skip Step 2 if we just want to transform data and tune models directly

## 3.1 Transformation Function for Data cleaning


```python
#import necessary packages here
import warnings
warnings.filterwarnings('ignore')
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from subprocess import check_output
# print(check_output(["ls", "../input"]).decode("utf8"))

np.random.seed(2020)
```


```python
def transform_data(data, song_df, members_df):
    # Merge song data with data set
    data = data.merge(song_df,on='song_id',how="left", copy =False)
    
    # preprocess member data
    members_df["registration_init_time"] = pd.to_datetime(members_df["registration_init_time"], format="%Y%m%d")
    members_df["expiration_date"] = pd.to_datetime(members_df["expiration_date"], format="%Y%m%d")
    members_df["registration_init_day"] = members_df["registration_init_time"].dt.day
    members_df["registration_init_month"] = members_df["registration_init_time"].dt.month
    members_df["registration_init_year"] = members_df["registration_init_time"].dt.year
    members_df["expiration_day"] = members_df["expiration_date"].dt.day
    members_df["expiration_month"] = members_df["expiration_date"].dt.month
    members_df["expiration_year"] = members_df["expiration_date"].dt.year
    members_df = members_df.drop(columns = ["registration_init_time", "expiration_date"],axis=1)    
    
    # merge member data with dataset
    data = data.merge(members_df,on='msno',how="left", copy=False)
    
    # Remove outliers of bd age 
    data['bd'].loc[(data['bd']<=0) | (data['bd']>=100)]= np.nan

    categorical_ls = ['source_system_tab', 'source_screen_name','source_type','genre_ids','artist_name','composer',
     'lyricist','gender']


    numerical_ls = ['song_length','language','bd']
    # Fill missing values 
    for index in numerical_ls:
        data[index].fillna(data[index].median(), inplace=True)
    for index in categorical_ls:
        data[index].fillna("no_data", inplace=True)
        
    
    def count_items(x):
        if x =="no_data":
            return 0
        return sum(map(x.count, ['|', '/', '\\', ';',','])) + 1

    data['genre_count']=  data['genre_ids'].apply(count_items)
    data['composer_count']=  data['composer'].apply(count_items)
    data['lyricist_count']=  data['lyricist'].apply(count_items)
    
    # Convert object type to categorical type
    for c in data.columns:
        if data[c].dtype=='O':
            data[c] = data[c].astype("category",copy=False)
    if 'id' in data.columns:
        ids = data['id']
        data.drop(['id'], inplace=True,axis=1)
    else:
        ids =None
    return ids, data
```


```python
root = './kaggle/working/train/data/'
train_df = pd.read_csv(root+ "train.csv")
test_df = pd.read_csv(root+ "test.csv")
song_df = pd.read_csv(root+ "songs.csv")
# song_extra_df = pd.read_csv(root+ "song_extra_info.csv")
members_df = pd.read_csv(root+ "members.csv")
```


```python
_, train_data = transform_data(train_df, song_df, members_df)
```

Visualize the Correlation matrix after transforming data


```python
corr_matrix = train_data.corr()
_ = sns.heatmap(corr_matrix)
```


![png](kkboxmusicrecommendation_notebook_v7_files/kkboxmusicrecommendation_notebook_v7_114_0.png)



```python
corr_matrix
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>target</th>
      <th>song_length</th>
      <th>language</th>
      <th>city</th>
      <th>bd</th>
      <th>registered_via</th>
      <th>registration_init_day</th>
      <th>registration_init_month</th>
      <th>registration_init_year</th>
      <th>expiration_day</th>
      <th>expiration_month</th>
      <th>expiration_year</th>
      <th>genre_count</th>
      <th>composer_count</th>
      <th>lyricist_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>target</th>
      <td>1.000000</td>
      <td>-0.001809</td>
      <td>-0.027537</td>
      <td>0.012114</td>
      <td>-0.029062</td>
      <td>0.009893</td>
      <td>-0.001765</td>
      <td>-0.005573</td>
      <td>-0.001962</td>
      <td>0.001923</td>
      <td>0.011817</td>
      <td>0.042248</td>
      <td>-0.005689</td>
      <td>0.027213</td>
      <td>0.043528</td>
    </tr>
    <tr>
      <th>song_length</th>
      <td>-0.001809</td>
      <td>1.000000</td>
      <td>-0.210436</td>
      <td>0.005185</td>
      <td>0.007685</td>
      <td>0.002749</td>
      <td>-0.000002</td>
      <td>0.001518</td>
      <td>-0.007435</td>
      <td>0.000385</td>
      <td>0.001309</td>
      <td>0.004572</td>
      <td>-0.026094</td>
      <td>-0.102079</td>
      <td>-0.015731</td>
    </tr>
    <tr>
      <th>language</th>
      <td>-0.027537</td>
      <td>-0.210436</td>
      <td>1.000000</td>
      <td>-0.022197</td>
      <td>-0.000909</td>
      <td>-0.000232</td>
      <td>0.001511</td>
      <td>-0.006916</td>
      <td>0.009072</td>
      <td>-0.000452</td>
      <td>-0.001253</td>
      <td>-0.003990</td>
      <td>0.077519</td>
      <td>0.362132</td>
      <td>-0.021813</td>
    </tr>
    <tr>
      <th>city</th>
      <td>0.012114</td>
      <td>0.005185</td>
      <td>-0.022197</td>
      <td>1.000000</td>
      <td>0.057512</td>
      <td>0.073756</td>
      <td>0.007842</td>
      <td>-0.033314</td>
      <td>-0.280175</td>
      <td>0.004892</td>
      <td>-0.024833</td>
      <td>0.150147</td>
      <td>-0.005528</td>
      <td>-0.004215</td>
      <td>0.004835</td>
    </tr>
    <tr>
      <th>bd</th>
      <td>-0.029062</td>
      <td>0.007685</td>
      <td>-0.000909</td>
      <td>0.057512</td>
      <td>1.000000</td>
      <td>0.162485</td>
      <td>0.006868</td>
      <td>0.001192</td>
      <td>-0.269699</td>
      <td>0.047448</td>
      <td>0.020474</td>
      <td>0.040012</td>
      <td>0.005317</td>
      <td>-0.035854</td>
      <td>-0.035338</td>
    </tr>
    <tr>
      <th>registered_via</th>
      <td>0.009893</td>
      <td>0.002749</td>
      <td>-0.000232</td>
      <td>0.073756</td>
      <td>0.162485</td>
      <td>1.000000</td>
      <td>0.025543</td>
      <td>-0.017697</td>
      <td>-0.442730</td>
      <td>0.056956</td>
      <td>0.064732</td>
      <td>0.084135</td>
      <td>-0.001597</td>
      <td>-0.010430</td>
      <td>-0.012037</td>
    </tr>
    <tr>
      <th>registration_init_day</th>
      <td>-0.001765</td>
      <td>-0.000002</td>
      <td>0.001511</td>
      <td>0.007842</td>
      <td>0.006868</td>
      <td>0.025543</td>
      <td>1.000000</td>
      <td>0.044437</td>
      <td>-0.047175</td>
      <td>0.149351</td>
      <td>-0.013526</td>
      <td>-0.002089</td>
      <td>0.000056</td>
      <td>0.000031</td>
      <td>0.000829</td>
    </tr>
    <tr>
      <th>registration_init_month</th>
      <td>-0.005573</td>
      <td>0.001518</td>
      <td>-0.006916</td>
      <td>-0.033314</td>
      <td>0.001192</td>
      <td>-0.017697</td>
      <td>0.044437</td>
      <td>1.000000</td>
      <td>-0.047701</td>
      <td>-0.006693</td>
      <td>0.056911</td>
      <td>-0.047909</td>
      <td>0.000605</td>
      <td>-0.004635</td>
      <td>-0.000032</td>
    </tr>
    <tr>
      <th>registration_init_year</th>
      <td>-0.001962</td>
      <td>-0.007435</td>
      <td>0.009072</td>
      <td>-0.280175</td>
      <td>-0.269699</td>
      <td>-0.442730</td>
      <td>-0.047175</td>
      <td>-0.047701</td>
      <td>1.000000</td>
      <td>-0.078462</td>
      <td>-0.057629</td>
      <td>-0.092018</td>
      <td>0.003892</td>
      <td>0.013826</td>
      <td>0.013835</td>
    </tr>
    <tr>
      <th>expiration_day</th>
      <td>0.001923</td>
      <td>0.000385</td>
      <td>-0.000452</td>
      <td>0.004892</td>
      <td>0.047448</td>
      <td>0.056956</td>
      <td>0.149351</td>
      <td>-0.006693</td>
      <td>-0.078462</td>
      <td>1.000000</td>
      <td>-0.029495</td>
      <td>-0.030104</td>
      <td>0.001198</td>
      <td>-0.002181</td>
      <td>-0.003404</td>
    </tr>
    <tr>
      <th>expiration_month</th>
      <td>0.011817</td>
      <td>0.001309</td>
      <td>-0.001253</td>
      <td>-0.024833</td>
      <td>0.020474</td>
      <td>0.064732</td>
      <td>-0.013526</td>
      <td>0.056911</td>
      <td>-0.057629</td>
      <td>-0.029495</td>
      <td>1.000000</td>
      <td>-0.472842</td>
      <td>0.000675</td>
      <td>-0.002876</td>
      <td>-0.002433</td>
    </tr>
    <tr>
      <th>expiration_year</th>
      <td>0.042248</td>
      <td>0.004572</td>
      <td>-0.003990</td>
      <td>0.150147</td>
      <td>0.040012</td>
      <td>0.084135</td>
      <td>-0.002089</td>
      <td>-0.047909</td>
      <td>-0.092018</td>
      <td>-0.030104</td>
      <td>-0.472842</td>
      <td>1.000000</td>
      <td>-0.004097</td>
      <td>-0.007370</td>
      <td>-0.008300</td>
    </tr>
    <tr>
      <th>genre_count</th>
      <td>-0.005689</td>
      <td>-0.026094</td>
      <td>0.077519</td>
      <td>-0.005528</td>
      <td>0.005317</td>
      <td>-0.001597</td>
      <td>0.000056</td>
      <td>0.000605</td>
      <td>0.003892</td>
      <td>0.001198</td>
      <td>0.000675</td>
      <td>-0.004097</td>
      <td>1.000000</td>
      <td>0.033796</td>
      <td>0.035215</td>
    </tr>
    <tr>
      <th>composer_count</th>
      <td>0.027213</td>
      <td>-0.102079</td>
      <td>0.362132</td>
      <td>-0.004215</td>
      <td>-0.035854</td>
      <td>-0.010430</td>
      <td>0.000031</td>
      <td>-0.004635</td>
      <td>0.013826</td>
      <td>-0.002181</td>
      <td>-0.002876</td>
      <td>-0.007370</td>
      <td>0.033796</td>
      <td>1.000000</td>
      <td>0.426352</td>
    </tr>
    <tr>
      <th>lyricist_count</th>
      <td>0.043528</td>
      <td>-0.015731</td>
      <td>-0.021813</td>
      <td>0.004835</td>
      <td>-0.035338</td>
      <td>-0.012037</td>
      <td>0.000829</td>
      <td>-0.000032</td>
      <td>0.013835</td>
      <td>-0.003404</td>
      <td>-0.002433</td>
      <td>-0.008300</td>
      <td>0.035215</td>
      <td>0.426352</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



**Analysis**: we can see that composer count is highly correlated to language, lyricist_count. The top 3 attributes correlated to target are lyricist_count, expiration year, language


```python
y_train = train_data['target']
train_data.drop(['target'], axis=1,inplace=True)
X_train = train_data
```


```python
X_train.head()
```

## 3.2 Transform features: composer, artist, lyricist, to create new features, like counts of composer, artist, lyricist, etc

### 3.2.1 Transform train set

Transform the name of  composer, artist, lyricist to new features like counts, number of intersection of names 


```python
def transform_names_intersection(data):
    #This function finds the intersection of names in composer, artist, lyricist
    def check_name_list(x):
        #convert string to name list dataframe
        strings = None
        strings = x.str.split(r"//|/|;|、|\| ")
        return strings
    
    df = data[["composer","artist_name", "lyricist"]].apply(check_name_list)
    data["composer_artist_intersect"] =[len(set(a) & set(b)) for a, b in zip(df.composer, df.artist_name)] 
    data["composer_lyricist_intersect"] =[len(set(a) & set(b)) for a, b in zip(df.composer, df.lyricist)] 
    data["artist_lyricist_intersect"] =[len(set(a) & set(b)) for a, b in zip(df.artist_name, df.lyricist)] 
    return data
    
_ = transform_names_intersection(X_train)
X_train.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>msno</th>
      <th>song_id</th>
      <th>source_system_tab</th>
      <th>source_screen_name</th>
      <th>source_type</th>
      <th>song_length</th>
      <th>genre_ids</th>
      <th>artist_name</th>
      <th>composer</th>
      <th>lyricist</th>
      <th>language</th>
      <th>city</th>
      <th>bd</th>
      <th>gender</th>
      <th>registered_via</th>
      <th>registration_init_day</th>
      <th>registration_init_month</th>
      <th>registration_init_year</th>
      <th>expiration_day</th>
      <th>expiration_month</th>
      <th>expiration_year</th>
      <th>genre_count</th>
      <th>composer_count</th>
      <th>lyricist_count</th>
      <th>composer_artist_intersect</th>
      <th>composer_lyricist_intersect</th>
      <th>artist_lyricist_intersect</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>FGtllVqz18RPiwJj/edr2gV78zirAiY/9SmYvia+kCg=</td>
      <td>BBzumQNXUHKdEBOB7mAJuzok+IJA1c2Ryg/yzTF6tik=</td>
      <td>explore</td>
      <td>Explore</td>
      <td>online-playlist</td>
      <td>206471.0</td>
      <td>359</td>
      <td>Bastille</td>
      <td>Dan Smith| Mark Crew</td>
      <td>no_data</td>
      <td>52.0</td>
      <td>1</td>
      <td>27.0</td>
      <td>no_data</td>
      <td>7</td>
      <td>2</td>
      <td>1</td>
      <td>2012</td>
      <td>5</td>
      <td>10</td>
      <td>2017</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Xumu+NIjS6QYVxDS4/t3SawvJ7viT9hPKXmf0RtLNx8=</td>
      <td>bhp/MpSNoqoxOIB+/l8WPqu6jldth4DIpCm3ayXnJqM=</td>
      <td>my library</td>
      <td>Local playlist more</td>
      <td>local-playlist</td>
      <td>284584.0</td>
      <td>1259</td>
      <td>Various Artists</td>
      <td>no_data</td>
      <td>no_data</td>
      <td>52.0</td>
      <td>13</td>
      <td>24.0</td>
      <td>female</td>
      <td>9</td>
      <td>25</td>
      <td>5</td>
      <td>2011</td>
      <td>11</td>
      <td>9</td>
      <td>2017</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Xumu+NIjS6QYVxDS4/t3SawvJ7viT9hPKXmf0RtLNx8=</td>
      <td>JNWfrrC7zNN7BdMpsISKa4Mw+xVJYNnxXh3/Epw7QgY=</td>
      <td>my library</td>
      <td>Local playlist more</td>
      <td>local-playlist</td>
      <td>225396.0</td>
      <td>1259</td>
      <td>Nas</td>
      <td>N. Jones、W. Adams、J. Lordan、D. Ingle</td>
      <td>no_data</td>
      <td>52.0</td>
      <td>13</td>
      <td>24.0</td>
      <td>female</td>
      <td>9</td>
      <td>25</td>
      <td>5</td>
      <td>2011</td>
      <td>11</td>
      <td>9</td>
      <td>2017</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Xumu+NIjS6QYVxDS4/t3SawvJ7viT9hPKXmf0RtLNx8=</td>
      <td>2A87tzfnJTSWqD7gIZHisolhe4DMdzkbd6LzO1KHjNs=</td>
      <td>my library</td>
      <td>Local playlist more</td>
      <td>local-playlist</td>
      <td>255512.0</td>
      <td>1019</td>
      <td>Soundway</td>
      <td>Kwadwo Donkoh</td>
      <td>no_data</td>
      <td>-1.0</td>
      <td>13</td>
      <td>24.0</td>
      <td>female</td>
      <td>9</td>
      <td>25</td>
      <td>5</td>
      <td>2011</td>
      <td>11</td>
      <td>9</td>
      <td>2017</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>FGtllVqz18RPiwJj/edr2gV78zirAiY/9SmYvia+kCg=</td>
      <td>3qm6XTZ6MOCU11x8FIVbAGH5l5uMkT3/ZalWG1oo2Gc=</td>
      <td>explore</td>
      <td>Explore</td>
      <td>online-playlist</td>
      <td>187802.0</td>
      <td>1011</td>
      <td>Brett Young</td>
      <td>Brett Young| Kelly Archer| Justin Ebach</td>
      <td>no_data</td>
      <td>52.0</td>
      <td>1</td>
      <td>27.0</td>
      <td>no_data</td>
      <td>7</td>
      <td>2</td>
      <td>1</td>
      <td>2012</td>
      <td>5</td>
      <td>10</td>
      <td>2017</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



### 3.2.2 Transform Testset


```python
ids, test_data = transform_data(test_df, song_df, members_df)
_ = transform_names_intersection(test_data)
test_data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>msno</th>
      <th>song_id</th>
      <th>source_system_tab</th>
      <th>source_screen_name</th>
      <th>source_type</th>
      <th>song_length</th>
      <th>genre_ids</th>
      <th>artist_name</th>
      <th>composer</th>
      <th>lyricist</th>
      <th>language</th>
      <th>city</th>
      <th>bd</th>
      <th>gender</th>
      <th>registered_via</th>
      <th>registration_init_day</th>
      <th>registration_init_month</th>
      <th>registration_init_year</th>
      <th>expiration_day</th>
      <th>expiration_month</th>
      <th>expiration_year</th>
      <th>genre_count</th>
      <th>composer_count</th>
      <th>lyricist_count</th>
      <th>composer_artist_intersect</th>
      <th>composer_lyricist_intersect</th>
      <th>artist_lyricist_intersect</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>V8ruy7SGk7tDm3zA51DPpn6qutt+vmKMBKa21dp54uM=</td>
      <td>WmHKgKMlp1lQMecNdNvDMkvIycZYHnFwDT72I5sIssc=</td>
      <td>my library</td>
      <td>Local playlist more</td>
      <td>local-library</td>
      <td>224130.0</td>
      <td>458</td>
      <td>梁文音 (Rachel Liang)</td>
      <td>Qi Zheng Zhang</td>
      <td>no_data</td>
      <td>3.0</td>
      <td>1</td>
      <td>27.0</td>
      <td>no_data</td>
      <td>7</td>
      <td>19</td>
      <td>2</td>
      <td>2016</td>
      <td>18</td>
      <td>9</td>
      <td>2017</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>V8ruy7SGk7tDm3zA51DPpn6qutt+vmKMBKa21dp54uM=</td>
      <td>y/rsZ9DC7FwK5F2PK2D5mj+aOBUJAjuu3dZ14NgE0vM=</td>
      <td>my library</td>
      <td>Local playlist more</td>
      <td>local-library</td>
      <td>320470.0</td>
      <td>465</td>
      <td>林俊傑 (JJ Lin)</td>
      <td>林俊傑</td>
      <td>孫燕姿/易家揚</td>
      <td>3.0</td>
      <td>1</td>
      <td>27.0</td>
      <td>no_data</td>
      <td>7</td>
      <td>19</td>
      <td>2</td>
      <td>2016</td>
      <td>18</td>
      <td>9</td>
      <td>2017</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>/uQAlrAkaczV+nWCd2sPF2ekvXPRipV7q0l+gbLuxjw=</td>
      <td>8eZLFOdGVdXBSqoAv5nsLigeH2BvKXzTQYtUM53I0k4=</td>
      <td>discover</td>
      <td>no_data</td>
      <td>song-based-playlist</td>
      <td>315899.0</td>
      <td>2022</td>
      <td>Yu Takahashi (高橋優)</td>
      <td>Yu Takahashi</td>
      <td>Yu Takahashi</td>
      <td>17.0</td>
      <td>1</td>
      <td>27.0</td>
      <td>no_data</td>
      <td>4</td>
      <td>17</td>
      <td>11</td>
      <td>2016</td>
      <td>24</td>
      <td>11</td>
      <td>2016</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1a6oo/iXKatxQx4eS9zTVD+KlSVaAFbTIqVvwLC1Y0k=</td>
      <td>ztCf8thYsS4YN3GcIL/bvoxLm/T5mYBVKOO4C9NiVfQ=</td>
      <td>radio</td>
      <td>Radio</td>
      <td>radio</td>
      <td>285210.0</td>
      <td>465</td>
      <td>U2</td>
      <td>The Edge| Adam Clayton| Larry Mullen| Jr.</td>
      <td>no_data</td>
      <td>52.0</td>
      <td>3</td>
      <td>30.0</td>
      <td>male</td>
      <td>9</td>
      <td>25</td>
      <td>7</td>
      <td>2007</td>
      <td>30</td>
      <td>4</td>
      <td>2017</td>
      <td>1</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1a6oo/iXKatxQx4eS9zTVD+KlSVaAFbTIqVvwLC1Y0k=</td>
      <td>MKVMpslKcQhMaFEgcEQhEfi5+RZhMYlU3eRDpySrH8Y=</td>
      <td>radio</td>
      <td>Radio</td>
      <td>radio</td>
      <td>197590.0</td>
      <td>873</td>
      <td>Yoga Mr Sound</td>
      <td>Neuromancer</td>
      <td>no_data</td>
      <td>-1.0</td>
      <td>3</td>
      <td>30.0</td>
      <td>male</td>
      <td>9</td>
      <td>25</td>
      <td>7</td>
      <td>2007</td>
      <td>30</td>
      <td>4</td>
      <td>2017</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
test_data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>msno</th>
      <th>song_id</th>
      <th>source_system_tab</th>
      <th>source_screen_name</th>
      <th>source_type</th>
      <th>song_length</th>
      <th>genre_ids</th>
      <th>artist_name</th>
      <th>composer</th>
      <th>lyricist</th>
      <th>language</th>
      <th>city</th>
      <th>bd</th>
      <th>gender</th>
      <th>registered_via</th>
      <th>registration_init_day</th>
      <th>registration_init_month</th>
      <th>registration_init_year</th>
      <th>expiration_day</th>
      <th>expiration_month</th>
      <th>expiration_year</th>
      <th>genre_count</th>
      <th>composer_count</th>
      <th>lyricist_count</th>
      <th>composer_artist_intersect</th>
      <th>composer_lyricist_intersect</th>
      <th>artist_lyricist_intersect</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>V8ruy7SGk7tDm3zA51DPpn6qutt+vmKMBKa21dp54uM=</td>
      <td>WmHKgKMlp1lQMecNdNvDMkvIycZYHnFwDT72I5sIssc=</td>
      <td>my library</td>
      <td>Local playlist more</td>
      <td>local-library</td>
      <td>224130.0</td>
      <td>458</td>
      <td>梁文音 (Rachel Liang)</td>
      <td>Qi Zheng Zhang</td>
      <td>no_data</td>
      <td>3.0</td>
      <td>1</td>
      <td>27.0</td>
      <td>no_data</td>
      <td>7</td>
      <td>19</td>
      <td>2</td>
      <td>2016</td>
      <td>18</td>
      <td>9</td>
      <td>2017</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>V8ruy7SGk7tDm3zA51DPpn6qutt+vmKMBKa21dp54uM=</td>
      <td>y/rsZ9DC7FwK5F2PK2D5mj+aOBUJAjuu3dZ14NgE0vM=</td>
      <td>my library</td>
      <td>Local playlist more</td>
      <td>local-library</td>
      <td>320470.0</td>
      <td>465</td>
      <td>林俊傑 (JJ Lin)</td>
      <td>林俊傑</td>
      <td>孫燕姿/易家揚</td>
      <td>3.0</td>
      <td>1</td>
      <td>27.0</td>
      <td>no_data</td>
      <td>7</td>
      <td>19</td>
      <td>2</td>
      <td>2016</td>
      <td>18</td>
      <td>9</td>
      <td>2017</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>/uQAlrAkaczV+nWCd2sPF2ekvXPRipV7q0l+gbLuxjw=</td>
      <td>8eZLFOdGVdXBSqoAv5nsLigeH2BvKXzTQYtUM53I0k4=</td>
      <td>discover</td>
      <td>no_data</td>
      <td>song-based-playlist</td>
      <td>315899.0</td>
      <td>2022</td>
      <td>Yu Takahashi (高橋優)</td>
      <td>Yu Takahashi</td>
      <td>Yu Takahashi</td>
      <td>17.0</td>
      <td>1</td>
      <td>27.0</td>
      <td>no_data</td>
      <td>4</td>
      <td>17</td>
      <td>11</td>
      <td>2016</td>
      <td>24</td>
      <td>11</td>
      <td>2016</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1a6oo/iXKatxQx4eS9zTVD+KlSVaAFbTIqVvwLC1Y0k=</td>
      <td>ztCf8thYsS4YN3GcIL/bvoxLm/T5mYBVKOO4C9NiVfQ=</td>
      <td>radio</td>
      <td>Radio</td>
      <td>radio</td>
      <td>285210.0</td>
      <td>465</td>
      <td>U2</td>
      <td>The Edge| Adam Clayton| Larry Mullen| Jr.</td>
      <td>no_data</td>
      <td>52.0</td>
      <td>3</td>
      <td>30.0</td>
      <td>male</td>
      <td>9</td>
      <td>25</td>
      <td>7</td>
      <td>2007</td>
      <td>30</td>
      <td>4</td>
      <td>2017</td>
      <td>1</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1a6oo/iXKatxQx4eS9zTVD+KlSVaAFbTIqVvwLC1Y0k=</td>
      <td>MKVMpslKcQhMaFEgcEQhEfi5+RZhMYlU3eRDpySrH8Y=</td>
      <td>radio</td>
      <td>Radio</td>
      <td>radio</td>
      <td>197590.0</td>
      <td>873</td>
      <td>Yoga Mr Sound</td>
      <td>Neuromancer</td>
      <td>no_data</td>
      <td>-1.0</td>
      <td>3</td>
      <td>30.0</td>
      <td>male</td>
      <td>9</td>
      <td>25</td>
      <td>7</td>
      <td>2007</td>
      <td>30</td>
      <td>4</td>
      <td>2017</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



## 3.3 Split validation set and trainset


```python
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold, train_test_split
ss_split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=2021)

# Split training set and testing set
train_index, valid_index ,test_index = None, None, None
for train_i, test_i in ss_split.split(np.zeros(y_train.shape) ,y_train):
    train_index = train_i
    test_index = test_i
    print("Train set size:",len(train_index), "Test set size:",len(test_index))
    


```

    (5901934,) (1475484,)
    


```python
X_validset = X_train.iloc[test_index]
y_validset = y_train.iloc[test_index].values

X_trainset = X_train.iloc[train_index]
y_trainset = y_train.iloc[train_index].values

#delete dataframes to save space
del X_train, y_train

```

Check the information of training set after splitting dataset


```python
X_trainset.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 5901934 entries, 7066318 to 5275539
    Data columns (total 27 columns):
     #   Column                       Dtype   
    ---  ------                       -----   
     0   msno                         category
     1   song_id                      category
     2   source_system_tab            category
     3   source_screen_name           category
     4   source_type                  category
     5   song_length                  float64 
     6   genre_ids                    category
     7   artist_name                  category
     8   composer                     category
     9   lyricist                     category
     10  language                     float64 
     11  city                         int64   
     12  bd                           float64 
     13  gender                       category
     14  registered_via               int64   
     15  registration_init_day        int64   
     16  registration_init_month      int64   
     17  registration_init_year       int64   
     18  expiration_day               int64   
     19  expiration_month             int64   
     20  expiration_year              int64   
     21  genre_count                  int64   
     22  composer_count               int64   
     23  lyricist_count               int64   
     24  composer_artist_intersect    int64   
     25  composer_lyricist_intersect  int64   
     26  artist_lyricist_intersect    int64   
    dtypes: category(10), float64(3), int64(14)
    memory usage: 966.0 MB
    

# 4. LGBM Modeling
Light Gradient Boosting Machine (LGBM) model is a tree-based model, which use Boosting ensemble learning method to combine and train multiple decision tree models to do classification or regression task. It can automatically encode categorical data into labels or one-hot encoding vectors and train models.

Since it provides a fast way to transform categorical data and train model with good accuracy performance based on the results from leaderboard in kaggle, we try it here and tune its parameters to fit the data. Here is the reference to use LGBM: https://lightgbm.readthedocs.io/en/latest/Quick-Start.html

**Input to models:**
 The input features to LGBM model are the features shown above

**Output from models:**
The output from LGBM model is the possibility that user may repeat listening to the music

**Parameter Settings:**
+ Objective loss function: binary cross entropy loss
+ metric: area under curve of False Positive Rate vs True Positive Rate (AUC)
+ Number of leaves: 110
+ max_depth: [10, 15, 20, 25, 30]
+ Boosting method: gradient boosting 
+ Number of rounds: 200
+ maximum bin: 256
+ bagging fraction: 0.95


```python
import lightgbm as lgb
train_set = lgb.Dataset(X_trainset, y_trainset)
valid_set = lgb.Dataset(X_validset, y_validset)
```


```python
num_leaves = 110
max_depths = [10, 15, 20, 25,30]
```

## 5.Model Training and Validation on LGBM models


```python
params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting': 'gbdt',
        'learning_rate': 0.3 ,
        'verbose': 0,
        'num_leaves': num_leaves,
        'bagging_fraction': 0.95,
        'bagging_freq': 1,
        'bagging_seed': 1,
        'feature_fraction': 0.9,
        'feature_fraction_seed': 1,
        'max_bin': 256,
        'max_depth': max_depths[0],
        'num_rounds': 200,
        'metric' : 'auc'
    }

%time model_f1 = lgb.train(params, train_set=train_set,  valid_sets=valid_set, verbose_eval=5)
```

    [5]	valid_0's auc: 0.710928
    [10]	valid_0's auc: 0.723954
    [15]	valid_0's auc: 0.731661
    [20]	valid_0's auc: 0.736653
    [25]	valid_0's auc: 0.740424
    [30]	valid_0's auc: 0.744678
    [35]	valid_0's auc: 0.749056
    [40]	valid_0's auc: 0.752277
    [45]	valid_0's auc: 0.754501
    [50]	valid_0's auc: 0.756448
    [55]	valid_0's auc: 0.758097
    [60]	valid_0's auc: 0.75991
    [65]	valid_0's auc: 0.761418
    [70]	valid_0's auc: 0.762683
    [75]	valid_0's auc: 0.764243
    [80]	valid_0's auc: 0.765646
    [85]	valid_0's auc: 0.766883
    [90]	valid_0's auc: 0.767921
    [95]	valid_0's auc: 0.769111
    [100]	valid_0's auc: 0.770006
    [105]	valid_0's auc: 0.770934
    [110]	valid_0's auc: 0.772012
    [115]	valid_0's auc: 0.772747
    [120]	valid_0's auc: 0.773835
    [125]	valid_0's auc: 0.774486
    [130]	valid_0's auc: 0.775258
    [135]	valid_0's auc: 0.775887
    [140]	valid_0's auc: 0.776838
    [145]	valid_0's auc: 0.777587
    [150]	valid_0's auc: 0.778113
    [155]	valid_0's auc: 0.778714
    [160]	valid_0's auc: 0.77929
    [165]	valid_0's auc: 0.779884
    [170]	valid_0's auc: 0.780354
    [175]	valid_0's auc: 0.781586
    [180]	valid_0's auc: 0.782002
    [185]	valid_0's auc: 0.782517
    [190]	valid_0's auc: 0.783075
    [195]	valid_0's auc: 0.783496
    [200]	valid_0's auc: 0.784083
    CPU times: user 8min 20s, sys: 3 s, total: 8min 23s
    Wall time: 4min 25s
    


```python
params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting': 'gbdt',
        'learning_rate': 0.3 ,
        'verbose': 0,
        'num_leaves': num_leaves,
        'bagging_fraction': 0.95,
        'bagging_freq': 1,
        'bagging_seed': 1,
        'feature_fraction': 0.9,
        'feature_fraction_seed': 1,
        'max_bin': 256,
        'max_depth': max_depths[1],
        'num_rounds': 200,
        'metric' : 'auc'
    }

%time model_f2 = lgb.train(params, train_set=train_set,  valid_sets=valid_set, verbose_eval=5)
```

    [5]	valid_0's auc: 0.727917
    [10]	valid_0's auc: 0.742629
    [15]	valid_0's auc: 0.74811
    [20]	valid_0's auc: 0.754257
    [25]	valid_0's auc: 0.758256
    [30]	valid_0's auc: 0.76119
    [35]	valid_0's auc: 0.763674
    [40]	valid_0's auc: 0.76626
    [45]	valid_0's auc: 0.7681
    [50]	valid_0's auc: 0.769933
    [55]	valid_0's auc: 0.771692
    [60]	valid_0's auc: 0.773121
    [65]	valid_0's auc: 0.774693
    [70]	valid_0's auc: 0.776149
    [75]	valid_0's auc: 0.777157
    [80]	valid_0's auc: 0.778674
    [85]	valid_0's auc: 0.780085
    [90]	valid_0's auc: 0.78098
    [95]	valid_0's auc: 0.782016
    [100]	valid_0's auc: 0.783028
    [105]	valid_0's auc: 0.783782
    [110]	valid_0's auc: 0.784875
    [115]	valid_0's auc: 0.785417
    [120]	valid_0's auc: 0.786042
    [125]	valid_0's auc: 0.78665
    [130]	valid_0's auc: 0.787237
    [135]	valid_0's auc: 0.787897
    [140]	valid_0's auc: 0.788426
    [145]	valid_0's auc: 0.788904
    [150]	valid_0's auc: 0.789517
    [155]	valid_0's auc: 0.78991
    [160]	valid_0's auc: 0.790561
    [165]	valid_0's auc: 0.791319
    [170]	valid_0's auc: 0.791855
    [175]	valid_0's auc: 0.792519
    [180]	valid_0's auc: 0.792922
    [185]	valid_0's auc: 0.793727
    [190]	valid_0's auc: 0.794061
    [195]	valid_0's auc: 0.794584
    [200]	valid_0's auc: 0.794811
    CPU times: user 9min 27s, sys: 1.28 s, total: 9min 28s
    Wall time: 4min 51s
    


```python
params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting': 'gbdt',
        'learning_rate': 0.3 ,
        'verbose': 0,
        'num_leaves': num_leaves,
        'bagging_fraction': 0.95,
        'bagging_freq': 1,
        'bagging_seed': 1,
        'feature_fraction': 0.9,
        'feature_fraction_seed': 1,
        'max_bin': 256,
        'max_depth': max_depths[2],
        'num_rounds': 200,
        'metric' : 'auc'
    }

%time model_f3 = lgb.train(params, train_set=train_set,  valid_sets=valid_set, verbose_eval=5)
```

    [5]	valid_0's auc: 0.734133
    [10]	valid_0's auc: 0.749742
    [15]	valid_0's auc: 0.75615
    [20]	valid_0's auc: 0.761276
    [25]	valid_0's auc: 0.766358
    [30]	valid_0's auc: 0.769127
    [35]	valid_0's auc: 0.771531
    [40]	valid_0's auc: 0.773761
    [45]	valid_0's auc: 0.775287
    [50]	valid_0's auc: 0.777329
    [55]	valid_0's auc: 0.779154
    [60]	valid_0's auc: 0.780391
    [65]	valid_0's auc: 0.782072
    [70]	valid_0's auc: 0.783786
    [75]	valid_0's auc: 0.784989
    [80]	valid_0's auc: 0.785685
    [85]	valid_0's auc: 0.786851
    [90]	valid_0's auc: 0.787643
    [95]	valid_0's auc: 0.788312
    [100]	valid_0's auc: 0.789305
    [105]	valid_0's auc: 0.790256
    [110]	valid_0's auc: 0.791037
    [115]	valid_0's auc: 0.79177
    [120]	valid_0's auc: 0.792466
    [125]	valid_0's auc: 0.792988
    [130]	valid_0's auc: 0.793478
    [135]	valid_0's auc: 0.793961
    [140]	valid_0's auc: 0.794871
    [145]	valid_0's auc: 0.795495
    [150]	valid_0's auc: 0.795952
    [155]	valid_0's auc: 0.796269
    [160]	valid_0's auc: 0.796888
    [165]	valid_0's auc: 0.797808
    [170]	valid_0's auc: 0.7982
    [175]	valid_0's auc: 0.798443
    [180]	valid_0's auc: 0.798959
    [185]	valid_0's auc: 0.799395
    [190]	valid_0's auc: 0.799687
    [195]	valid_0's auc: 0.800153
    [200]	valid_0's auc: 0.800409
    CPU times: user 11min 16s, sys: 1.54 s, total: 11min 17s
    Wall time: 5min 47s
    


```python
params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting': 'gbdt',
        'learning_rate': 0.3 ,
        'verbose': 0,
        'num_leaves': num_leaves,
        'bagging_fraction': 0.95,
        'bagging_freq': 1,
        'bagging_seed': 1,
        'feature_fraction': 0.9,
        'feature_fraction_seed': 1,
        'max_bin': 256,
        'max_depth': max_depths[3],
        'num_rounds': 200,
        'metric' : 'auc'
    }

%time model_f4 = lgb.train(params, train_set=train_set,  valid_sets=valid_set, verbose_eval=5)
```

    [5]	valid_0's auc: 0.736351
    [10]	valid_0's auc: 0.754592
    [15]	valid_0's auc: 0.76195
    [20]	valid_0's auc: 0.766405
    [25]	valid_0's auc: 0.770538
    [30]	valid_0's auc: 0.772566
    [35]	valid_0's auc: 0.775789
    [40]	valid_0's auc: 0.777994
    [45]	valid_0's auc: 0.779658
    [50]	valid_0's auc: 0.781394
    [55]	valid_0's auc: 0.783194
    [60]	valid_0's auc: 0.784808
    [65]	valid_0's auc: 0.786109
    [70]	valid_0's auc: 0.787265
    [75]	valid_0's auc: 0.788079
    [80]	valid_0's auc: 0.789109
    [85]	valid_0's auc: 0.78986
    [90]	valid_0's auc: 0.790613
    [95]	valid_0's auc: 0.791347
    [100]	valid_0's auc: 0.79209
    [105]	valid_0's auc: 0.793348
    [110]	valid_0's auc: 0.79409
    [115]	valid_0's auc: 0.794754
    [120]	valid_0's auc: 0.795411
    [125]	valid_0's auc: 0.795866
    [130]	valid_0's auc: 0.796604
    [135]	valid_0's auc: 0.79781
    [140]	valid_0's auc: 0.798172
    [145]	valid_0's auc: 0.798723
    [150]	valid_0's auc: 0.799132
    [155]	valid_0's auc: 0.799488
    [160]	valid_0's auc: 0.800115
    [165]	valid_0's auc: 0.800509
    [170]	valid_0's auc: 0.800784
    [175]	valid_0's auc: 0.801118
    [180]	valid_0's auc: 0.801448
    [185]	valid_0's auc: 0.801882
    [190]	valid_0's auc: 0.8022
    [195]	valid_0's auc: 0.802578
    [200]	valid_0's auc: 0.802953
    CPU times: user 12min 34s, sys: 1.44 s, total: 12min 36s
    Wall time: 6min 27s
    


```python
params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting': 'gbdt',
        'learning_rate': 0.3 ,
        'verbose': 0,
        'num_leaves': num_leaves,
        'bagging_fraction': 0.95,
        'bagging_freq': 1,
        'bagging_seed': 1,
        'feature_fraction': 0.9,
        'feature_fraction_seed': 1,
        'max_bin': 256,
        'max_depth': max_depths[4],
        'num_rounds': 200,
        'metric' : 'auc'
    }

%time model_f5 = lgb.train(params, train_set=train_set,  valid_sets=valid_set, verbose_eval=5)
```

    [5]	valid_0's auc: 0.739442
    [10]	valid_0's auc: 0.757442
    [15]	valid_0's auc: 0.766439
    [20]	valid_0's auc: 0.77132
    [25]	valid_0's auc: 0.774735
    [30]	valid_0's auc: 0.777071
    [35]	valid_0's auc: 0.779247
    [40]	valid_0's auc: 0.781616
    [45]	valid_0's auc: 0.782953
    [50]	valid_0's auc: 0.785154
    [55]	valid_0's auc: 0.786877
    [60]	valid_0's auc: 0.787993
    [65]	valid_0's auc: 0.788839
    [70]	valid_0's auc: 0.790254
    [75]	valid_0's auc: 0.791088
    [80]	valid_0's auc: 0.792455
    [85]	valid_0's auc: 0.79365
    [90]	valid_0's auc: 0.794445
    [95]	valid_0's auc: 0.795072
    [100]	valid_0's auc: 0.796276
    [105]	valid_0's auc: 0.797737
    [110]	valid_0's auc: 0.798265
    [115]	valid_0's auc: 0.799021
    [120]	valid_0's auc: 0.799964
    [125]	valid_0's auc: 0.800469
    [130]	valid_0's auc: 0.801445
    [135]	valid_0's auc: 0.801851
    [140]	valid_0's auc: 0.802299
    [145]	valid_0's auc: 0.802599
    [150]	valid_0's auc: 0.803381
    [155]	valid_0's auc: 0.803696
    [160]	valid_0's auc: 0.803926
    [165]	valid_0's auc: 0.80443
    [170]	valid_0's auc: 0.804694
    [175]	valid_0's auc: 0.804897
    [180]	valid_0's auc: 0.80524
    [185]	valid_0's auc: 0.805486
    [190]	valid_0's auc: 0.805804
    [195]	valid_0's auc: 0.806059
    [200]	valid_0's auc: 0.806525
    CPU times: user 14min 7s, sys: 1.66 s, total: 14min 9s
    Wall time: 7min 14s
    

## 6.Model Evaluation on LGBM models


```python
from sklearn.metrics import accuracy_score
def evaluation_lgbm(model, X =X_validset , y= y_validset):
    
    out = model.predict(X)
    preds = out>=0.5
    acc = accuracy_score(preds, y)
    print("Evaluation acc:", acc)
    return acc

```


```python
X_validset.shape

```




    (1475484, 27)




```python
acc_1 = evaluation_lgbm(model_f1)
acc_2 = evaluation_lgbm(model_f2)
acc_3 = evaluation_lgbm(model_f3)
acc_4 = evaluation_lgbm(model_f4)
acc_5 = evaluation_lgbm(model_f5)
```

    Evaluation acc: 0.709764389176704
    Evaluation acc: 0.719106408473423
    Evaluation acc: 0.7236893114395005
    Evaluation acc: 0.7258221708944319
    Evaluation acc: 0.728842196865571
    


```python
eval_df = pd.DataFrame({"Lgbm with max_depth":max_depths,"Validation Accuracy":[acc_1,acc_2,acc_3,acc_4,acc_5]})
eval_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Lgbm with max_depth</th>
      <th>Validation Accuracy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>10</td>
      <td>0.709764</td>
    </tr>
    <tr>
      <th>1</th>
      <td>15</td>
      <td>0.719106</td>
    </tr>
    <tr>
      <th>2</th>
      <td>20</td>
      <td>0.723689</td>
    </tr>
    <tr>
      <th>3</th>
      <td>25</td>
      <td>0.725822</td>
    </tr>
    <tr>
      <th>4</th>
      <td>30</td>
      <td>0.728842</td>
    </tr>
  </tbody>
</table>
</div>



## Create Submission Files


```python
models = [model_f1,model_f2,model_f3,model_f4,model_f5]
for i in range(len(models)):
  preds_test = models[i].predict(test_data)
  submission = pd.DataFrame()
  submission['id'] = ids
  submission['target'] = preds_test
  submission.to_csv(root + 'submission_lgbm_model_'+ str(i)+'.csv.gz', compression = 'gzip', index=False, float_format = '%.5f')
  print("Predictions from model ",i,": ",preds_test)
```

    Predictions from model  0 :  [0.47177512 0.48584262 0.19651648 ... 0.39917036 0.30263348 0.36468783]
    Predictions from model  1 :  [0.45280296 0.55415074 0.17824637 ... 0.41500494 0.30757934 0.34520384]
    Predictions from model  2 :  [0.39847416 0.48724786 0.15954141 ... 0.38293317 0.27657349 0.28451098]
    Predictions from model  3 :  [0.3825275  0.39659855 0.15904321 ... 0.3515784  0.21812496 0.28995803]
    Predictions from model  4 :  [0.3951268  0.45704878 0.14609333 ... 0.35033303 0.23065677 0.2885925 ]
    

## Accuracy Scores from kaggle test set

|Model name       | private score  | public score |
|- |-| -|
| LGBM Boosting Machine Model 4 | 0.67423 | 0.67256
| LGBM Boosting Machine Model 3 | 0.67435 | 0.67241
| LGBM Boosting Machine Model 2 | 0.67416 | 0.67208
| LGBM Boosting Machine Model 1 | 0.67416 | 0.67188
| LGBM Boosting Machine Model 0 | 0.67206 | 0.66940

# 4. Wide & Depth neural network model

### Label Encoding for categorical data
Convert categorical data into numerical labels before using embedding


```python
from sklearn.preprocessing import LabelEncoder
categorical_ls1 = ['source_system_tab', 'source_screen_name','source_type','genre_ids','gender']
categorical_ls2 = ['artist_name','composer', 'lyricist']


numerical_ls = ['song_length','language','bd',"registration_init_year",
                "expiration_day","expiration_month","expiration_year",
                "genre_count","composer_count","lyricist_count","composer_artist_intersect",
                "composer_lyricist_intersect","artist_lyricist_intersect"]

max_values = {}
# labelencoders = {}
for col in categorical_ls1:
    print(col)
    lbl = LabelEncoder()
    df = pd.concat([X_trainset[col], X_validset[col],test_data[col]],ignore_index=True)
    lbl.fit(df)
    df = lbl.transform(list(df.values.astype('str',copy=False)))
    X_trainset[col] = lbl.transform(list(X_trainset[col].values.astype('str',copy=False)))
    X_validset[col] = lbl.transform(list(X_validset[col].values.astype('str',copy=False)))
    test_data[col] = lbl.transform(list(test_data[col].values.astype('str',copy=False)))
    max_values[col] = df.max() + 2 #set the range of embedding input larger

# Compute embedding dimensions
emb_dims1 = []
emb_dims2 = []
for i in categorical_ls1:
    emb_dims1.append((max_values[i], min((max_values[i]+1)//2, 50)))
```

    source_system_tab
    source_screen_name
    source_type
    genre_ids
    gender
    


```python
# max_values
X_trainset.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>msno</th>
      <th>song_id</th>
      <th>source_system_tab</th>
      <th>source_screen_name</th>
      <th>source_type</th>
      <th>song_length</th>
      <th>genre_ids</th>
      <th>artist_name</th>
      <th>composer</th>
      <th>lyricist</th>
      <th>language</th>
      <th>city</th>
      <th>bd</th>
      <th>gender</th>
      <th>registered_via</th>
      <th>registration_init_day</th>
      <th>registration_init_month</th>
      <th>registration_init_year</th>
      <th>expiration_day</th>
      <th>expiration_month</th>
      <th>expiration_year</th>
      <th>genre_count</th>
      <th>composer_count</th>
      <th>lyricist_count</th>
      <th>composer_artist_intersect</th>
      <th>composer_lyricist_intersect</th>
      <th>artist_lyricist_intersect</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>7066318</th>
      <td>BQ7nOoOUipsqjOBANK+ilA8F7TVaOHSI8gVPWElXsuI=</td>
      <td>FaTUlIiCh/6sEOasPm1vgIk9XqavgSGgRGYuOkzTF0o=</td>
      <td>0</td>
      <td>3</td>
      <td>7</td>
      <td>203520.0</td>
      <td>364</td>
      <td>田馥甄 (Hebe)</td>
      <td>倪子岡</td>
      <td>李格弟</td>
      <td>3.0</td>
      <td>5</td>
      <td>25.0</td>
      <td>0</td>
      <td>9</td>
      <td>2</td>
      <td>9</td>
      <td>2006</td>
      <td>11</td>
      <td>10</td>
      <td>2017</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1471565</th>
      <td>Ul+UpO5PxuhCn040AK8gzR1A/mE/k3KbL13gO7Uc4Ts=</td>
      <td>+SstqMwhQPBQFTPBhLKPT642IiBDXzZFwlzsLl4cGXo=</td>
      <td>3</td>
      <td>8</td>
      <td>3</td>
      <td>283846.0</td>
      <td>371</td>
      <td>陳勢安 (Andrew Tan)</td>
      <td>覃嘉健</td>
      <td>馬嵩惟</td>
      <td>3.0</td>
      <td>13</td>
      <td>47.0</td>
      <td>0</td>
      <td>9</td>
      <td>1</td>
      <td>2</td>
      <td>2006</td>
      <td>30</td>
      <td>9</td>
      <td>2017</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6176886</th>
      <td>sa6oKy94c62R5Eq0YHkNzZrJSo9j5E7JGjTDHnYRKqs=</td>
      <td>K6fBQxiNhgWazjXrZUGlZIm9ltT4o+Vq19sWmZRdAhg=</td>
      <td>2</td>
      <td>12</td>
      <td>2</td>
      <td>296960.0</td>
      <td>371</td>
      <td>蔡依林 (Jolin Tsai)</td>
      <td>郭子</td>
      <td>鄔裕康</td>
      <td>3.0</td>
      <td>5</td>
      <td>38.0</td>
      <td>1</td>
      <td>7</td>
      <td>27</td>
      <td>11</td>
      <td>2011</td>
      <td>30</td>
      <td>12</td>
      <td>2017</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3527889</th>
      <td>LG/BLgJxw5AvXy0pkgaHYYWeU7jKS+ms/51+7TaBY9Y=</td>
      <td>O+/KJ5a5GzbgLZrCOw/t/iDOPTrDcrz5ZnOtaK9blA8=</td>
      <td>3</td>
      <td>8</td>
      <td>0</td>
      <td>235403.0</td>
      <td>200</td>
      <td>ONE OK ROCK</td>
      <td>Toru/Taka</td>
      <td>Taka</td>
      <td>17.0</td>
      <td>1</td>
      <td>27.0</td>
      <td>2</td>
      <td>7</td>
      <td>1</td>
      <td>6</td>
      <td>2013</td>
      <td>1</td>
      <td>10</td>
      <td>2017</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6073849</th>
      <td>KmAJtsNcrofH6qMoHvET89mQAlC1EN3r3r3rkfW2iT4=</td>
      <td>WogFv1yz1n49l4gNSbf76bWxas8nNvzHntrj4FuzC24=</td>
      <td>3</td>
      <td>22</td>
      <td>7</td>
      <td>210604.0</td>
      <td>371</td>
      <td>Twins</td>
      <td>no_data</td>
      <td>no_data</td>
      <td>24.0</td>
      <td>13</td>
      <td>28.0</td>
      <td>1</td>
      <td>9</td>
      <td>2</td>
      <td>11</td>
      <td>2016</td>
      <td>7</td>
      <td>10</td>
      <td>2017</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
X_validset.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>msno</th>
      <th>song_id</th>
      <th>source_system_tab</th>
      <th>source_screen_name</th>
      <th>source_type</th>
      <th>song_length</th>
      <th>genre_ids</th>
      <th>artist_name</th>
      <th>composer</th>
      <th>lyricist</th>
      <th>language</th>
      <th>city</th>
      <th>bd</th>
      <th>gender</th>
      <th>registered_via</th>
      <th>registration_init_day</th>
      <th>registration_init_month</th>
      <th>registration_init_year</th>
      <th>expiration_day</th>
      <th>expiration_month</th>
      <th>expiration_year</th>
      <th>genre_count</th>
      <th>composer_count</th>
      <th>lyricist_count</th>
      <th>composer_artist_intersect</th>
      <th>composer_lyricist_intersect</th>
      <th>artist_lyricist_intersect</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3400479</th>
      <td>RZDFiPWpvwi1RWF5NAPEkvmogqe+7rGys+zoLU9he2M=</td>
      <td>MvON55vzjT7QW7GSs/UVLZrE/LJpMAVFUjXwZczdw40=</td>
      <td>0</td>
      <td>11</td>
      <td>7</td>
      <td>356379.0</td>
      <td>97</td>
      <td>Nas</td>
      <td>Amy Winehouse| Salaam Remi| Nasir Jones p/k/a NAS</td>
      <td>no_data</td>
      <td>52.0</td>
      <td>1</td>
      <td>27.0</td>
      <td>2</td>
      <td>7</td>
      <td>7</td>
      <td>12</td>
      <td>2010</td>
      <td>20</td>
      <td>9</td>
      <td>2017</td>
      <td>1</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2481022</th>
      <td>C3EZ5oh7XDt5fP9OY20RPlD8MA+rBknmvmDhA1tHGMU=</td>
      <td>5PvPCUIB7vVuCNpQRKXIOcWvh9EerujDAbrjV7G6ZE0=</td>
      <td>3</td>
      <td>8</td>
      <td>3</td>
      <td>216920.0</td>
      <td>349</td>
      <td>貴族精選</td>
      <td>no_data</td>
      <td>Super Market| Microdot</td>
      <td>31.0</td>
      <td>5</td>
      <td>27.0</td>
      <td>2</td>
      <td>3</td>
      <td>11</td>
      <td>12</td>
      <td>2012</td>
      <td>5</td>
      <td>1</td>
      <td>2018</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5808216</th>
      <td>O1pwjdTED6P3lKm52VBxVUtaSVc31S9PmIw+07WBNw4=</td>
      <td>va3+1L2wraJkzDbHjvdo+e+0TTJcLko0k0pqBn09nJE=</td>
      <td>3</td>
      <td>8</td>
      <td>3</td>
      <td>268225.0</td>
      <td>548</td>
      <td>Various Artists</td>
      <td>no_data</td>
      <td>no_data</td>
      <td>3.0</td>
      <td>13</td>
      <td>82.0</td>
      <td>0</td>
      <td>9</td>
      <td>29</td>
      <td>4</td>
      <td>2007</td>
      <td>23</td>
      <td>1</td>
      <td>2018</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>42686</th>
      <td>WFCCMzA4hADGBduTS6X8mXlutyiC0P33QkTG6zr5yCg=</td>
      <td>U9kojfZSKaiWOW94PKh1Riyv/zUWxmBRmv0XInQWLGw=</td>
      <td>7</td>
      <td>11</td>
      <td>7</td>
      <td>290063.0</td>
      <td>364</td>
      <td>周杰倫 (Jay Chou)</td>
      <td>周杰倫</td>
      <td>方文山</td>
      <td>3.0</td>
      <td>13</td>
      <td>32.0</td>
      <td>0</td>
      <td>7</td>
      <td>12</td>
      <td>12</td>
      <td>2010</td>
      <td>9</td>
      <td>9</td>
      <td>2017</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1850837</th>
      <td>h0fTru8nYMv9bR0j6kBh8kiXDaybzWBYaSHbUIVzeBs=</td>
      <td>J1sgBEFbcXSK6eiN7CK1WNxsso0/sY6t0BMX+c+iPNw=</td>
      <td>0</td>
      <td>11</td>
      <td>7</td>
      <td>220450.0</td>
      <td>111</td>
      <td>Usher</td>
      <td>no_data</td>
      <td>no_data</td>
      <td>52.0</td>
      <td>22</td>
      <td>28.0</td>
      <td>0</td>
      <td>9</td>
      <td>2</td>
      <td>1</td>
      <td>2008</td>
      <td>28</td>
      <td>9</td>
      <td>2017</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



### **Wide and Deep Neural network model（2 branches-->merge two branches-->main branch）**
![网络图](https://github.com/LianxinGao/kaggle/blob/master/pics/net.jpg?raw=true)

This model transform categorical attributes into dense vectors using embedding network in neural network, which enable us to reduce the dimension of categorical data and extract main features like PCA.

Then it combines dense embedded vectors with numerical data for features selection and classifcation in the main branch. The main branch is a traditional Neural network using linear layers, activation functions (relu, softmax, sigmoid),which works as a classfication function

The output is possibility that user may repeat listening to the music.

#### **Network Architecture**
The architecture of neural network is as follow:
+ Block 1:
  + Linear
  + Relu
  + Dropout
  + Batch Normalization

+ Branch of categorical data:
  + categorical input data
  + Embedding
  + Block1
  + Block1

+ Main Branch:
  + Concatenate output from Branch of categorical data and Numerical data
  + Block1 
  + Block1
  + Linear
  + Softmax


The architecture of Wide and Deep Model is referred to https://github.com/zenwan/Wide-and-Deep-PyTorch

We also modify the output layer of the model to predict possibility value in range [0,1] using sigmoid function and use one branch of categorical data only.

#### **Input Features to network**
The input features to category embedding branch are: 
  + source_system_tab , source_screen_name ,source_type ,genre_ids ,gender

The input features to numerical branch are:
 + song_length, language , bd/age, registration_init_year, expiration_day, expiration_month,expiration_year, genre_count,composer_count,lyricist_count,composer_artist_intersect, composer_lyricist_intersect,artist_lyricist_intersect

  


```python
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
class TabularDataset(Dataset):
    def __init__(self, x_data, y_data, cat_cols1, cat_cols2, num_cols):
        
        """
        data: pandas data frame;
        cat_cols: list of string, the names of the categorical columns in the data, will be passed through the embedding layers;
        num_cols: list of string
        y_data: the target
        """
        self.n = x_data.shape[0]
        self.y = y_data.astype(np.float32).reshape(-1, 1)#.values.reshape(-1, 1)
       
        self.cat_cols1 = cat_cols1
        self.cat_cols2 = cat_cols2
        self.num_cols = num_cols
        
        self.num_X = x_data[self.num_cols].astype(np.float32).values
        self.cat_X1 = x_data[self.cat_cols1].astype(np.int64).values
        self.cat_X2 = x_data[self.cat_cols2].astype(np.int64).values
        
    
    def print_data(self):
        return self.num_X, self.cat_X1, self.cat_X2, self.y
    
    def __len__(self):
        """
        total number of samples
        """
        return self.n
    
    def __getitem__(self, idx):
        """
        Generates one sample of data.
        """
        return [self.y[idx], self.num_X[idx], self.cat_X1[idx], self.cat_X2[idx]]


```


```python
train_dataset = TabularDataset(x_data=X_trainset, y_data=y_trainset, cat_cols1=categorical_ls1, 
                               cat_cols2=[], num_cols=numerical_ls)

val_dataset = TabularDataset(x_data=X_validset, y_data=y_validset, cat_cols1=categorical_ls1,
                             cat_cols2=[], num_cols=numerical_ls)


```


```python
class FeedForwardNN(nn.Module):
    def __init__(self, emb_dims1, emb_dims2, no_of_num, lin_layer_sizes, output_size, emb_dropout, lin_layer_dropouts, branch2_enable=0):
        """
        emb_dims:           List of two element tuples;
        no_of_num:          Integer, the number of continuous features in the data;
        lin_layer_sizes:    List of integers. The size of each linear layer;
        output_size:        Integer, the size of the final output;
        emb_dropout:        Float, the dropout to be used after the embedding layers.
        lin_layer_dropouts: List of floats, the dropouts to be used after each linear layer.
        """
        super().__init__()
        self.branch2_enable = branch2_enable
        # embedding layers
        self.emb_layers1 = nn.ModuleList([nn.Embedding(x, y) for x, y in emb_dims1])
        self.emb_layers2 = nn.ModuleList([nn.Embedding(x, y) for x, y in emb_dims2])
        
        # 计算各个emb参数数量，为后续Linear layer的输入做准备
        self.no_of_embs1 = sum([y for x, y in emb_dims1])
        self.no_of_embs2 = sum([y for x, y in emb_dims2])
        self.no_of_num = no_of_num
        
        # 分支1
        self.branch1 = nn.Linear(self.no_of_embs1, lin_layer_sizes[0])
        self.branch1_2 = nn.Linear(lin_layer_sizes[0], lin_layer_sizes[1])
        nn.init.kaiming_normal_(self.branch1.weight.data)
        nn.init.kaiming_normal_(self.branch1_2.weight.data)
        
        # 分支2
        if branch2_enable:
            self.branch2 = nn.Linear(self.no_of_embs2, lin_layer_sizes[0] * 2)
            self.branch2_2 = nn.Linear(lin_layer_sizes[0] * 2, lin_layer_sizes[1] * 2)
            nn.init.kaiming_normal_(self.branch2.weight.data)
            nn.init.kaiming_normal_(self.branch2_2.weight.data)

        # 主分支
#         self.main_layer1 = nn.Linear(lin_layer_sizes[1] * 3 + self.no_of_num, lin_layer_sizes[2])
        self.main_layer1 = nn.Linear(77, lin_layer_sizes[2])
        self.main_layer2 = nn.Linear(lin_layer_sizes[2], lin_layer_sizes[3])
        
        # batch normal
        self.branch_bn_layers1 = nn.BatchNorm1d(lin_layer_sizes[0])
        self.branch_bn_layers2 = nn.BatchNorm1d(lin_layer_sizes[0] * 2)
        self.main_bn_layer = nn.BatchNorm1d(lin_layer_sizes[2])
        
        # Dropout Layers
        self.emb_dropout_layer = nn.Dropout(emb_dropout)
        self.dropout_layers = nn.ModuleList([nn.Dropout(size) for size in lin_layer_dropouts])
        
        # Output layer
        self.output_layer = nn.Linear(lin_layer_sizes[-1], output_size)
        nn.init.kaiming_normal_(self.output_layer.weight.data)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, num_data, cat_data1, cat_data2):
        # embedding categorical feature and cat them together
        x1 = [emb_layer(torch.tensor(cat_data1[:, i])) for i, emb_layer in enumerate(self.emb_layers1)]
        x1 = torch.cat(x1, 1)
        
        x1 = self.emb_dropout_layer(F.relu(self.branch1(x1)))
        x1 = self.branch_bn_layers1(x1)
        x1 = self.dropout_layers[0](F.relu(self.branch1_2(x1)))
        if self.branch2_enable:
            x2 = [emb_layer(torch.tensor(cat_data2[:, i])) for i, emb_layer in enumerate(self.emb_layers2)]
            x2 = torch.cat(x2, 1)

            x2 = self.emb_dropout_layer(F.relu(self.branch2(x2)))
            x2 = self.branch_bn_layers2(x2)
            x2 = self.dropout_layers[0](F.relu(self.branch2_2(x2)))

            main = torch.cat([x1, x2, num_data], 1)
        else:
            main = torch.cat([x1, num_data], 1)
#         print("Main Shape: ", main.shape)
        main = self.dropout_layers[1](F.relu(self.main_layer1(main)))
        main = self.main_bn_layer(main)
        main = self.dropout_layers[2](F.relu(self.main_layer2(main)))

        out = self.output_layer(main)
        out = self.sigmoid(out)
        return out
```


```python
batchsize = 64
train_dataloader = DataLoader(train_dataset, batchsize, shuffle=True, num_workers=2)
val_dataloader = DataLoader(val_dataset, 64, shuffle=True, num_workers=2)
```


```python
# next(iter(train_dataloader))[3]
```


```python
np.random.seed(2020)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FeedForwardNN(emb_dims1=emb_dims1,
                      emb_dims2=emb_dims2, 
                      no_of_num=len(numerical_ls),
                      lin_layer_sizes=[128,64,32,16],
                      output_size=1,
                      lin_layer_dropouts=[0.1, 0.1, 0.05],
                      emb_dropout=0.05).to(device)
```


```python
device,len(train_dataloader)
```




    (device(type='cpu'), 92218)



## 5.Model Training and Validation on Wide and Deep model


```python
no_of_epochs = 2
batch_num = 4000
# criterion = torch.nn.MSELoss()
criterion = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
lrscheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=2, threshold=0.9 )
total_data = train_dataset.__len__()
best_val_score = 0.0
best_model =None

print_every = 500
steps = 0
running_loss = 0

for epoch in range(no_of_epochs):
    model.train()
    batch_cnt = 0
    for index, datas in enumerate(train_dataloader):
        if batch_cnt == batch_num:
            break
        steps += 1
        batch_cnt += 1
        y, num_x, cat_x1, cat_x2 = datas
        cat_x1 = cat_x1.to(device)
        cat_x2 = cat_x2.to(device)
        num_x = num_x.to(device)
        y  = y.to(device)
        
        # Forward Pass
        optimizer.zero_grad()
        preds = model.forward(num_x, cat_x1, cat_x2)
        loss = criterion(preds, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
        if steps % print_every == 0:
            val_loss = 0
            model.eval()
            val_acc = 0.
            total_len = 0.
            with torch.no_grad():
                for val_index, val_datas in enumerate(val_dataloader):
                    y, num_x, cat_x1, cat_x2 = val_datas
                    cat_x1 = cat_x1.to(device)
                    cat_x2 = cat_x2.to(device)
                    num_x = num_x.to(device)
                    y  = y.to(device)
                    
                    out = model.forward(num_x, cat_x1, cat_x2)
                    batch_loss = criterion(out, y)
                    val_acc += ((out>0.5)==y ).sum().detach().to('cpu').numpy()
                    total_len += len(out)
                    
                    val_loss += batch_loss.item()

            val_acc /= total_len
            if val_acc> best_val_score:
                  best_val_score = val_acc
                  torch.save(model,"checkpoint.pt")
                      
                      # print("Checkpoint saved.")
            # update scheduler
            lrscheduler.step(val_loss)
            print(f"Epoch {epoch+1}/{no_of_epochs}.."
                     f"Train loss:{running_loss/print_every:.4f}.."
                     f"Validation loss:{val_loss/len(val_dataloader):.4f}.."
                 f"Validation Acc:{val_acc:.4f}.."
                 f"Best Validation Acc:{best_val_score:.4f}..")
            running_loss = 0
            model.train()
print("Training Completed")
best_model = torch.load("checkpoint.pt")
```

    Epoch 1/2..Train loss:0.6945..Validation loss:0.6925..Validation Acc:0.5302..Best Validation Acc:0.5302..
    Epoch 1/2..Train loss:0.6742..Validation loss:0.6663..Validation Acc:0.6229..Best Validation Acc:0.6229..
    Epoch 1/2..Train loss:0.6640..Validation loss:0.6618..Validation Acc:0.6245..Best Validation Acc:0.6245..
    Epoch 1/2..Train loss:0.6623..Validation loss:0.6768..Validation Acc:0.6252..Best Validation Acc:0.6252..
    Epoch 1/2..Train loss:0.6628..Validation loss:0.6648..Validation Acc:0.6249..Best Validation Acc:0.6252..
    Epoch 1/2..Train loss:0.6636..Validation loss:0.6656..Validation Acc:0.6253..Best Validation Acc:0.6253..
    Epoch 1/2..Train loss:0.6623..Validation loss:0.6674..Validation Acc:0.6254..Best Validation Acc:0.6254..
    Epoch 1/2..Train loss:0.6640..Validation loss:0.6649..Validation Acc:0.6255..Best Validation Acc:0.6255..
    Epoch 2/2..Train loss:0.6597..Validation loss:0.6638..Validation Acc:0.6256..Best Validation Acc:0.6256..
    Epoch 2/2..Train loss:0.6605..Validation loss:0.6584..Validation Acc:0.6259..Best Validation Acc:0.6259..
    Epoch 2/2..Train loss:0.6609..Validation loss:0.6636..Validation Acc:0.6263..Best Validation Acc:0.6263..
    Epoch 2/2..Train loss:0.6602..Validation loss:0.6620..Validation Acc:0.6263..Best Validation Acc:0.6263..
    Epoch 2/2..Train loss:0.6647..Validation loss:0.6658..Validation Acc:0.6264..Best Validation Acc:0.6264..
    Epoch 2/2..Train loss:0.6615..Validation loss:0.6644..Validation Acc:0.6262..Best Validation Acc:0.6264..
    Epoch 2/2..Train loss:0.6641..Validation loss:0.6764..Validation Acc:0.6254..Best Validation Acc:0.6264..
    Epoch 2/2..Train loss:0.6600..Validation loss:0.6926..Validation Acc:0.6263..Best Validation Acc:0.6264..
    Training Completed
    


```python
print(f"Best Validation Acc:{best_val_score:.4f}..")
```

    Best Validation Acc:0.6264..
    


```python
model = torch.load("checkpoint.pt")
```


```python
test_dataset = TabularDataset(x_data=test_data, y_data=np.zeros(len(test_data)), cat_cols1=categorical_ls1,
                             cat_cols2=[], num_cols=numerical_ls)
```


```python
def evaluation(test_dataloder):
    
    model.eval()
    total_cnt = 0.
    correct_cnt = 0.
    acc = None
    with torch.no_grad():
        for test_index, test_datas in enumerate(test_dataloder):
                        y, num_x, cat_x1, cat_x2 = test_datas
                        cat_x1 = cat_x1.to(device)
                        cat_x2 = cat_x2.to(device)
                        num_x = num_x.to(device)
                        y  = y.to(device)            
                        out = model.forward(num_x, cat_x1, cat_x2)
                        correct_cnt += ((out>0.5)==y ).sum().detach().to('cpu').numpy()
                        total_cnt += len(out)
#                         out = out.squeeze().to("cpu").numpy().tolist()
                        
        acc = 100* correct_cnt / total_cnt
        print("Evaluation Acc: %.4f %%"%(acc))
    return acc

```

## 6.Model Evaluation on Wide and Deep model using validation set


```python
acc = evaluation(val_dataloader)
```

    Evaluation Acc: 62.6432 %
    


```python
def predict_test(test_dataset):
    preds = []
    model.eval()
    test_dataloder = DataLoader(test_dataset, 200, shuffle=False, num_workers=4)
    with torch.no_grad():
        for test_index, test_datas in enumerate(test_dataloder):
                        y, num_x, cat_x1, cat_x2 = test_datas
                        cat_x1 = cat_x1.to(device)
                        cat_x2 = cat_x2.to(device)
                        num_x = num_x.to(device)
                        y  = y.to(device)            
                        out = model.forward(num_x, cat_x1, cat_x2)
                        out = out.squeeze().to("cpu").numpy().tolist()
#                         print(out)
                        preds.extend(out)
                        
    return np.array(preds)
        
```

## Make Predictions and submission


```python
preds = predict_test(test_dataset)
submission = pd.DataFrame()
submission['id'] = ids
submission['target'] = preds
submission.to_csv(root + 'submission_WideAndDeep_model.csv.gz', compression = 'gzip', index=False, float_format = '%.5f')
print("Model Predictions: ",preds)
# !kaggle competitions submit -c ./train/data -f submission_lgbm_model.csv.gz -m "Message"

```

    Model Predictions:  [0.60901934 0.60901934 0.35112065 ... 0.54463851 0.48204085 0.54300648]
    

submission_WideAndDeep_model.csv.gz
WideAndDeep_model
0.61628
0.61117


```python
perf_df = pd.DataFrame({"model name":['Wide and Deep model'],"private_score":[0.61628], "public score": [0.61117]})
perf_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>model name</th>
      <th>private_score</th>
      <th>public score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Wide and Deep model</td>
      <td>0.61628</td>
      <td>0.61117</td>
    </tr>
  </tbody>
</table>
</div>




```python
model_names = ["lgbm model 0","lgbm model 1","lgbm model 2","lgbm model 3","lgbm model 4"]
private_score = [0.67206,0.67416,0.67416,0.67435,0.67423]
public_score = [0.66940,0.67188,0.67208,0.67241,0.67256]
perf_df = pd.DataFrame({"model name":model_names,"max_depth:":max_depths,"private_score":private_score, "public score": public_score})
perf_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>model name</th>
      <th>max_depth:</th>
      <th>private_score</th>
      <th>public score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>lgbm model 0</td>
      <td>10</td>
      <td>0.67206</td>
      <td>0.66940</td>
    </tr>
    <tr>
      <th>1</th>
      <td>lgbm model 1</td>
      <td>15</td>
      <td>0.67416</td>
      <td>0.67188</td>
    </tr>
    <tr>
      <th>2</th>
      <td>lgbm model 2</td>
      <td>20</td>
      <td>0.67416</td>
      <td>0.67208</td>
    </tr>
    <tr>
      <th>3</th>
      <td>lgbm model 3</td>
      <td>25</td>
      <td>0.67435</td>
      <td>0.67241</td>
    </tr>
    <tr>
      <th>4</th>
      <td>lgbm model 4</td>
      <td>30</td>
      <td>0.67423</td>
      <td>0.67256</td>
    </tr>
  </tbody>
</table>
</div>



From the result above, we can see the best accuracy score on testset from kaggle is 67.256%. Note that in Kaggle leaderboard, the best score is 74% only. The preprocessing and transformation of dataset is still one of the keys to improve the performance. Good data transformation and creating new features  could be a way to improve the accuracy.

# 7. KKBox-Music Recommendation System Project Report

## Motivation
A recommendation system is a subclass of an information filtering system that seeks to predict the "rating" or "preference" a user would give to an item.
In Music Recommendation System, due to the fact that there are millions of musics, composers, artist and other information, it is hard for users to pick the musics they like from millions of songs. Hence it is necessary to build a music recommendation system to recommend musics, which users prefer and are likely to repeat listening to.

## Goal of this project
In this project, we are going to build a recommendation system to predict the chances of a user listening to a song repetitively after the first observable listening event within a time window was triggered. If there are recurring listening event(s) triggered within a month after the user’s very first observable listening event, its target is marked 1, and 0 otherwise in the training set. The same rule applies to the testing set.

## Data Collection and Description
In data collection, we introduce two ways to download the large dataset 1.7GB (which can not be stored in github so we put it in google drive): One is use Google Drive Link, the other one is to download it from kaggle website.
And the KKBox dataset is composed of following files:

1. train.csv
    + msno: user id
    + song_id: song id
    + source_system_tab: the name of the tab where the event was triggered. System tabs are used to categorize KKBOX mobile apps functions. For example, tab my library contains functions to manipulate the local storage, and tab search contains functions relating to search.
    + source_screen_name: name of the layout a user sees.
    + source_type: an entry point a user first plays music on mobile apps. An entry point could be album, online-playlist, song .. etc.
    + target: this is the target variable. target=1 means there are recurring listening event(s) triggered within a month after the user’s very first observable listening event, target=0 otherwise .
    
 
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

6. song_extra_info.csv
    + song_id
    + song name - the name of the song.
    + isrc - International Standard Recording Code, theoretically can be used as an identity of a song. However, what worth to note is, ISRCs generated from providers have not been officially verified; therefore the information in ISRC, such as country code and reference year, can be misleading/incorrect. Multiple songs could share one ISRC since a single recording could be re-published several times.

## Exploratory Data Analysis (EDA)
In data exploratory data analysis, we do the following visualization and analysis to explore the relationship between different pairs of attributes:

+ Find the Description and summary of each CSV file and determine Null object, categorical attributes, numerical attributes
+ Analyze Song information
  - Plot and visualize the count of song genres, composers, artists
  From the plots of genres, composers, artists, we find that such kinds of features are long-tailed distribution, with most of users focusing on listening to some specific genres of songs, or songs from specific composers and artists.

  - Plot and visualize the count of source types, source screen names, system tabs with target as hue axis. From the bar plot of source, screen names, system tabs, we find that most of users prefer repeating listening to music from their local library or local resources. Hence it is good to keep these kinds of features to train model later


+ Analyze Member information
  - visualize the count of bd/age attribute
  - visualize correlation between different attributes using heatmap plot

+ Plot bivariate plots to visualize and analyze relationship between attributes, like city and age/bd, expiration date and target. In this part of analysis, we can see the ages of most of users who repeat listening to songs are around 20~30.


## Data Preprocessing 
**Note that** This section is to preprocess data like filling missing values and removing outliers. 

**In order to train models, you should start from step 3 ETL to extract and transform data directly using integrated functions**

In Data Preprocessing step, we do the following to clean the data
+ Convert String datetime data to datetime format and separate year, month, day as new features
+ Remove the outlier in bd/age features and replace them with NaN.

+ Filling missing values for categorical data using "no_data" label and numerical data using median values

+ Adding new features, like the counts of composers, artists, lyricist, the counts of intersection names among composers, artists and lyricists.

+ Convert features from object type into categorical type.

## Data Pipeline: Extract, Transformation, Load (ETL) 
In this step, we integrate the data cleaning steps into transformation functions to clean and transform the data so that we can easily clean the data. Then we split the dataset into training set (80%) and validation set (20%) to train the models later. 

As the training set is used to train the models, the validation set is used to test and keep track of the accuracy performance of models on unseen data during training. Two dataset should be separated from each other.


## Machine Learning Modeling
+ **Task**
  Since this project is to predict the chances of a user listening to a song repetitively after the first observable listening event within a time window was triggered, the output from machine learning model is the possibility that a user will repeat listening to a song after the first listening event.
  Hence, this problem can be modeled as a binary classification problem

+ **Loss Function**
  Since this task is a binary classification problem, we simply select binary cross entropy loss as the loss function used to train parametric machine learning model, like deep learning model, logistic regression, etc.

+ **Evaluation Metric**
  The evaluation metric to measure the performance of machine learning model is selected to be accuracy and AUC (area under curve), since we care about how accurate the models could be..

+ **Light Gradient Boosting machines (LGBM) model**
  In the modeling part, we use LGBM model first, which is a light gradient boosting machine model using tree-based basic models for boosting. LGBM is one of ensemble learning methods, which uses multiple decision tree models for boosting to better fit the data and improve the accuracy performance.

  Since the dataset is large, 1.7 GB and the number of attributes can increase during data transformation and LGBM provides a very fast way to train a machine model and also achieve good accuracy performances based on previous works from other kaggle users,  we try LGBM models here.

  In this part, we train LGBM models with different max_depths of tree: [10, 15,20, 25, 30] to see how max_depth affects the accuracy on prediction

+ **Wide and Deep Neural network model**

  In addition to LGBM model, we are also interested in trying the Wide and Deep Neural network model since it is one of the popular neural network models in recommendation system and we want to see if this can help us improve the accuracy. The reference to Wide and Deep model is as follow: https://github.com/zenwan/Wide-and-Deep-PyTorch/blob/master/wide_deep/torch_model.py

  In wide and deep model, It first uses a technique called embedding, which projects the sparse categorical features into dense features vectors with smaller dimension and extract the main features. Then it concatenates the embedded vectors with the numerical features together to train a traditional neural network classifier.
  Before we train the Wide and Deep model, we use label encoder from sklearn to transform categorical data into numerical data. Then we use those numerical labels to train our network.

## Model Training and validation
+ In model training and validation step, we split the data set into training set(80% of dataset) and validation set (20% of dataset) and then use them to train and keep track of the performance of models. While the training set is used to train models, the validation set which is invisible to models is used to test the performance of models on unseen data, so that we can know if the model is overfitting or not. If training accuracy is increasing, but validation accuracy is decreasing, then the model is actually overfitting on training set

Another way to estimate the performance of models is Cross-Validation. However, since this method requires us to repeat training different models using different folds of dataset and the dataset is 1.7GB, it will be time-consuming to use Cross-validation on a large dataset. Hence, in this case, we simply use a hold-out dataset as validation dataset to validate model during training

## Model Evaluation and Results
In Model evaluation step, we simply use the validation dataset to validate the final trained models and then let models make predictions on testset from kaggle and submit predictions to kaggle to see the final evaluation scores.

In the final results on test data from kaggle, we can see that light gradient boosting machines have the accuracy performance better than Wide and Deep Neural Network model. The best score in LGBM models is 67.256% while the Wide and Deep model has accuracy of 61.11%. 

Moreover, we can observe that as the value of max_depth of decision tree in Boosting machine increases, both validation accuracy and teset accuracy increase gradually. It implies that the performance of our LGBM models may be improved by increasing the max_depth. As increasing max_depth can improve the learning/fitting ability of LGBM model, it is possible that tuning other parameters like number of leaves, number of training epoches may also help improve the accuracy and let models better fit the dataset.

Although the Wide and Deep model performance is not so well, we may improve its performance by tunning the parameters like dropout rate in neural network, learning rate, training epoches in the future. In addition, in our experiments we  try two epoches only, this is because we run the program in Google Colab and also try it in Kaggle platform, but the hardware is not powerful enough to train the model quickly and there is time limit in using GPU. Therefore, we can try better hardware to boost the training process in the future.


## Future Work
In this project, there are several things we can improve in the future:
+ We can use better hardware for training models, rather than using Google Colab or Kaggle platform, so that we can better train the deep learning model
+ Tune the LGBM models using grid search and choose larger max_depth values or tune other parameters
+ Try to create more new features from text attributes like composer, lyricist, artist and use feature importance methods to pick features that most contribute to prediction
+ Try a better neural network architecture to fit the data as we can see the network model here doesn't fit the dataset very well

## Summary 
In conclusion, we collect 1.7GB KKBOX music dataset from kaggle and do exploratory data analysis (EDA) on the data by visualizing the attributes and compute the correlations among features. Then we clean the dataset by removing outliers from age/bd attributes, filling missing categorical data with new label and missing numerical data with median value. We also transform the text data and create new features. After that we use 80% dataset as training set and 20% dataset as validation set.

In Modeling and Evaluation, we use LGBM models and Wide & Deep Neural network model to fit the dataset and also tune the max_depth parameter in LGBM to do binary classification task. The best accuracy performance of our models is 67.25% while the best accuracy from kaggle leaderboard is about 74%.

Lesson we learn from this project is that data analysis and transformation is one of the key parts to improve the performance of machine learning model, since it tells us the correlation between features and target and see which features contribute to the important information used to train models. In this section, we can either use visualization method, like count plot, distribution plot, etc to analyze the relationship between features and target, or use quantitaive measurement like correlation matrix, feature importance from decision tree, hypothesis testing, etc to determine the most important features.

In modeling parts, we can choose different models based on the task we do and properties of dataset. For example, in this music dataset, the features are not linear separated, in this case, we use LGBM model gradient boosting machine based on decision tree models (non-linear model) to fit the data. In addition, since the dataset contains many categorical data, we can use embedding method in deep learning to convert sparse categorical data into dense numerical data to reduce data dimension and train model. Since the prediction task is binary classification and we only care accuracy, we only use accuracy as metric  to evaluate performance of models.

At the end, we also summarize the future works to improve the project, like using better hardware resources, tunning other parameters of models and explore more useful features for training.



## Reference
[1] https://github.com/zenwan/Wide-and-Deep-PyTorch/blob/master/wide_deep/torch_model.py

[2] https://www.kaggle.com/c/kkbox-music-recommendation-challenge/submit

[3] https://www.kaggle.com/asmitavikas/feature-engineered-0-68310

[4] https://blog.statsbot.co/recommendation-system-algorithms-ba67f39ac9a3

[5] https://medium.com/@pushkarmandot/https-medium-com-pushkarmandot-what-is-lightgbm-how-to-implement-it-how-to-fine-tune-the-parameters-60347819b7fc

[6] https://towardsdatascience.com/how-to-build-a-wide-and-deep-model-using-keras-in-tensorflow-2-0-2f7a236b5a4b

[7] https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.Dataset.html#lightgbm.Dataset

