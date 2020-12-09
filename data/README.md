# KKBox's Music Recommendation Challenage Dataset
We collect the 1.7GB dataset from Kaggle KKBox's Music Recommendation Challenage. Since the dataset is large, we don't post it on Github
You can find the dataset from Kaggle: https://www.kaggle.com/c/kkbox-music-recommendation-challenge/data
Or you can get the dataset from our Google Drive: https://drive.google.com/file/d/1-WJHZUWFtz9ksfvFoX-dc-ZKjZ6fTk0D/view?usp=sharing

The KKBox dataset is composed of the following files:

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
