import requests
import json
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import time
from prepare import basic_clean, tokenize, lemmatize, stem, remove_stopwords, prep_repo_data, split_data
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from acquire import get_df

from wordcloud import WordCloud
from matplotlib import pyplot as plt
import seaborn as sns

import varname
import nltk

from scipy import stats

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score


import warnings
warnings.filterwarnings("ignore")



# def idf(word):
#     '''A simple way to calculate idf.'''
#     n_occurences = sum([1 for doc in repos if word in doc])
#     return len(repos) / n_occurences

def explore(df, train, validate, test):
    '''
    This function performs all exploration found in wheeler_notebook.ipynb.
    Creates TF-IDF, Analyzes word count, character count, 
    sentiment values, and other features fed to models.
    '''
    ### IDF

    def idf(word):
        '''A simple way to calculate idf.'''
        n_occurences = sum([1 for doc in repos if word in doc])
        return len(repos) / n_occurences
    repos = [repo for repo in train.lemmatized]
    unique_words = pd.Series(' '.join(repos).split()).unique()
    # put the unique words into a data frame
    words_idf = (pd.DataFrame(dict(word=unique_words))
        # calculate the idf for each word
        .assign(idf=lambda train: train.word.apply(idf))
        # sort the data for presentation purposes
        .set_index('word')
        .sort_values(by='idf', ascending=False))
    # create more stopwords
    more_stopwords = words_idf[(words_idf.idf <= 5)].index.to_list()

    # loop through each column in train
    for col in ['clean','stemmed','lemmatized']:
        # apply remove_stopwords function to columns in train
        train[col] = train[col].apply(remove_stopwords, extra_words = more_stopwords)
    
        # repeat for validate and test
        ##### More stopwords was retrieved from the test dataset so no data leakage occurs with this transformation
        validate[col] = validate[col].apply(remove_stopwords, extra_words = more_stopwords)
        test[col] = test[col].apply(remove_stopwords, extra_words = more_stopwords)
    
    #### Word Counts / Message Length

    # Create word count and message length columns for all datasets
    train['message_length'] = train.lemmatized.apply(len)
    train['word_count'] = train.lemmatized.apply(basic_clean).apply(str.split).apply(len)

    validate['message_length'] = validate.lemmatized.apply(len)
    validate['word_count'] = validate.lemmatized.apply(basic_clean).apply(str.split).apply(len)

    test['message_length'] = test.lemmatized.apply(len)
    test['word_count'] = test.lemmatized.apply(basic_clean).apply(str.split).apply(len)


    #### Sentiment
    # sia = nltk.sentiment.SentimentIntensityAnalyzer()
    # sia.polarity_scores(all_words)
    # train['sentiment'] = train.lemmatized.apply(lambda doc: sia.polarity_scores(doc)['compound'])
    # validate['sentiment'] = validate.lemmatized.apply(lambda doc: sia.polarity_scores(doc)['compound'])
    # test['sentiment'] = test.lemmatized.apply(lambda doc: sia.polarity_scores(doc)['compound'])


    ##### TF-IDF
    # redefine repos
    repos = [repo for repo in train.lemmatized]

    # create tfidf object
    tfidf = TfidfVectorizer()

    # fit/use tfidf
    #train
    tfidfs = tfidf.fit_transform(repos)

    #validate
    validate_tfidf = tfidf.transform([repo for repo in validate.lemmatized])

    #test
    test_tfidf = tfidf.transform([repo for repo in test.lemmatized])

    # create not-sparse matrices
    tfidf_df = pd.DataFrame(tfidfs.todense(), columns=tfidf.get_feature_names())
    validate_tfidf_df = pd.DataFrame(validate_tfidf.todense(), columns=tfidf.get_feature_names())
    test_tfidf_df = pd.DataFrame(test_tfidf.todense(), columns=tfidf.get_feature_names())

    # merge matrices to dataframes
    train = pd.merge(train,tfidf_df,how='left',right_index=True, left_index=True).fillna(0.0).rename(columns={'language_x':'coding_language', 'clean_x':'cleaned_readme', 'link_x':'repo_link'})
    validate = pd.merge(validate,tfidf_df,how='left',right_index=True, left_index=True).fillna(0.0).rename(columns={'language_x':'coding_language', 'clean_x':'cleaned_readme', 'link_x':'repo_link'})
    test = pd.merge(test,tfidf_df,how='left',right_index=True, left_index=True).fillna(0.0).rename(columns={'language_x':'coding_language', 'clean_x':'cleaned_readme', 'link_x':'repo_link'})
    return df, train, validate, test


def get_idf_dist(train):
    # create word strings
    javascript_words = basic_clean(' '.join(train[train.coding_language == 'JavaScript'].lemmatized))
    c_sharp_words = basic_clean(' '.join(train[train.coding_language == 'C#'].lemmatized))
    php_words = basic_clean(' '.join(train[train.coding_language == 'PHP'].lemmatized))
    c_words = basic_clean(' '.join(train[train.coding_language == 'C'].lemmatized))
    sourcepawn_words = basic_clean(' '.join(train[train.coding_language == 'SourcePawn'].lemmatized))
    html_words = basic_clean(' '.join(train[train.coding_language == 'HTML'].lemmatized))
    c_plus_plus_words = basic_clean(' '.join(train[train.coding_language == 'C++'].lemmatized))
    java_words = basic_clean(' '.join(train[train.coding_language == 'Java'].lemmatized))
    python_words = basic_clean(' '.join(train[train.coding_language == 'Python'].lemmatized))
    lua_words = basic_clean(' '.join(train[train.coding_language == 'Lua'].lemmatized))
    ruby_words = basic_clean(' '.join(train[train.coding_language == 'Ruby'].lemmatized))
    all_words = basic_clean(' '.join(train.lemmatized))

    # create series of word frequencies per language
    javascript_freq = pd.Series(javascript_words.split()).value_counts()
    c_sharp_freq = pd.Series(c_sharp_words.split()).value_counts()
    php_freq = pd.Series(php_words.split()).value_counts()
    c_freq = pd.Series(c_words.split()).value_counts()
    sourcepawn_freq = pd.Series(sourcepawn_words.split()).value_counts()
    html_freq = pd.Series(html_words.split()).value_counts()
    c_plus_plus_freq =pd.Series(c_plus_plus_words.split()).value_counts() 
    java_freq = pd.Series(java_words.split()).value_counts()
    python_freq = pd.Series(python_words.split()).value_counts()
    lua_freq = pd.Series(lua_words.split()).value_counts()
    ruby_freq = pd.Series(ruby_words.split()).value_counts()

    # all languages word frequency
    all_freq = pd.Series(all_words.split()).value_counts()

    word_counts = pd.concat([javascript_freq, c_sharp_freq, php_freq, c_freq, sourcepawn_freq, html_freq, c_plus_plus_freq, java_freq, python_freq, lua_freq, ruby_freq, all_freq], axis=1).fillna(0).astype(int)
    word_counts.columns = ['javascript', 'c_sharp', 'php','c','sourcepawn','html','c_plus_plus','java','python','lua','ruby','all']
    dist_loop = word_counts.sort_values('all', ascending = False).head(20).index.to_list()
    return dist_loop

