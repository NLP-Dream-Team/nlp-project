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
import nltk.sentiment

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


    ### Sentiment
    all_words = basic_clean(' '.join(train.lemmatized))
    sia = nltk.sentiment.SentimentIntensityAnalyzer()
    sia.polarity_scores(all_words)
    train['sentiment'] = train.lemmatized.apply(lambda doc: sia.polarity_scores(doc)['compound'])
    validate['sentiment'] = validate.lemmatized.apply(lambda doc: sia.polarity_scores(doc)['compound'])
    test['sentiment'] = test.lemmatized.apply(lambda doc: sia.polarity_scores(doc)['compound'])


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

def distributions_grid(df, quant_vars):

    '''
    This function creates a nice sized figure, enumerates the list of features passed into the function, creates a grid of subplots, and then charts histograms for features in the list onto the subplots.
    '''

    plt.figure(figsize = (13, 8), edgecolor = 'darkslategrey')   # create figure
    for i, cat in enumerate(quant_vars[:6]):    # loop through enumerated list
        plot_number = i + 1     # i starts at 0, but plot nos should start at 1
        plt.subplot(2, 3, plot_number)    # create subplot
        plt.title(cat)    # title 
        plt.ylabel('Count')    # set y-axis label
        plt.xlabel('Inverse Document Frequency')    # set x-axis label
        df[cat].hist(color = 'teal', edgecolor = 'plum')   # display histogram for column
        plt.grid(False)    # rid grid-lines
        plt.tight_layout();    # clean
    
def bar_messages(df):
    
    '''
    This function sets default figure and font sizes, groups the data by languages, and makes a dataframe
    out of the average message lengths. It then plots a bar chart using pandas to show average message length
    for each language.
    '''
    
    plt.rc('font', size = 15)    # set default text size
    plt.rc('figure', figsize = (13, 8))    # set default figure size

    # assign variables to respective data
    c_sharp = df[df.coding_language == 'C#']
    c = df[df.coding_language == 'C']
    c_plus_plus = df[df.coding_language == 'C++']
    html = df[df.coding_language == 'HTML']
    java = df[df.coding_language == 'Java']
    java_script = df[df.coding_language == 'JavaScript']
    lua = df[df.coding_language == 'Lua']
    php = df[df.coding_language == 'PHP']
    python = df[df.coding_language == 'Python']
    ruby = df[df.coding_language == 'Ruby']
    source_pawn = df[df.coding_language == 'SourcePawn']

    # list coding languages
    programs = list(df.coding_language.unique())
    # sort languages
    programs.sort()
    
    # assign dataframe to list of average message lengths
    unread_messages = pd.DataFrame([c.message_length.mean(),
                                    c_sharp.message_length.mean(),
                                    c_plus_plus.message_length.mean(),
                                    html.message_length.mean(),
                                    java.message_length.mean(),
                                    java_script.message_length.mean(),
                                    lua.message_length.mean(),
                                    php.message_length.mean(),
                                    python.message_length.mean(),
                                    ruby.message_length.mean(),
                                    source_pawn.message_length.mean()
                                   ], index = programs
                                  )

    unread_messages.columns = {'avg_message_length'}    # re-define column

    # plot bar chart of average message lengths
    unread_messages.plot.barh(color = 'darkmagenta', edgecolor = 'paleturquoise')
    plt.title('Average Message Length by Coding Language', size = 18)    # set title
    plt.xlabel('Character Count', size = 15)    # set x-axis label
    plt.ylabel('Language', size = 15);    # set y-axis label
     
def run_kruskal_wallis(train):
    '''
    This function runs a Kruskal-Wallis statistical test to determine if there is
    significant variation in the length of the README files based on what language
    the repository primarily contains.
    H_0: Average message length for languages is about the same.
    H_a: Average message length between languages is significantly different.
    '''
    # assign variables to respective data
    c_sharp = train[train.coding_language == 'C#']
    c = train[train.coding_language == 'C']
    c_plus_plus = train[train.coding_language == 'C++']
    html = train[train.coding_language == 'HTML']
    java = train[train.coding_language == 'Java']
    java_script = train[train.coding_language == 'JavaScript']
    lua = train[train.coding_language == 'Lua']
    php = train[train.coding_language == 'PHP']
    python = train[train.coding_language == 'Python']
    ruby = train[train.coding_language == 'Ruby']
    source_pawn = train[train.coding_language == 'SourcePawn']
    # set alpha
    alpha = 0.1
    # run kruskal-wallis test
    k_stat, p = stats.kruskal(c_sharp.message_length,
                              c.message_length,
                              c_plus_plus.message_length,
                              html.message_length,
                              java.message_length,
                              java_script.message_length,
                              lua.message_length,
                              php.message_length,
                              python.message_length,
                              ruby.message_length, 
                              source_pawn.message_length
                             )
    print(k_stat, p)
    # return result of test
    if p > alpha:
        return 'We fail to reject the null hypothesis.'
    else:
        return 'Reject the null hypothesis.'

def run_kruskal_wallis_sentiment(train):
    '''
    This function runs a Kruskal-Wallis statistical test to determine if there is
    significant variation in the sentiment of the README files based on what language
    the repository primarily contains.
    H_0: Average sentiment for languages is about the same.
    H_a: Average sentiment between languages is significantly different.
    '''
    # assign variables to respective data
    c_sharp = train[train.coding_language == 'C#']
    c = train[train.coding_language == 'C']
    c_plus_plus = train[train.coding_language == 'C++']
    html = train[train.coding_language == 'HTML']
    java = train[train.coding_language == 'Java']
    java_script = train[train.coding_language == 'JavaScript']
    lua = train[train.coding_language == 'Lua']
    php = train[train.coding_language == 'PHP']
    python = train[train.coding_language == 'Python']
    ruby = train[train.coding_language == 'Ruby']
    source_pawn = train[train.coding_language == 'SourcePawn']
    # set alpha
    alpha = 0.1
    # run kruskal-wallis test
    k_stat, p = stats.kruskal(c_sharp.sentiment,
                              c.sentiment,
                              c_plus_plus.sentiment,
                              html.sentiment,
                              java.sentiment,
                              java_script.sentiment,
                              lua.sentiment,
                              php.sentiment,
                              python.sentiment,
                              ruby.sentiment, 
                              source_pawn.sentiment
                             )
    print(k_stat, p)
    # return result of test
    if p > alpha:
        return 'We fail to reject the null hypothesis.'
    else:
        return 'Reject the null hypothesis.'

def sentiment_viz(train):
    '''
    This function returns a series of plots showing the distribution of sentiment scores by
    coding language.
    '''
    # assign variables to respective data
    c_sharp = train[train.coding_language == 'C#']
    c = train[train.coding_language == 'C']
    c_plus_plus = train[train.coding_language == 'C++']
    html = train[train.coding_language == 'HTML']
    java = train[train.coding_language == 'Java']
    java_script = train[train.coding_language == 'JavaScript']
    lua = train[train.coding_language == 'Lua']
    php = train[train.coding_language == 'PHP']
    python = train[train.coding_language == 'Python']
    ruby = train[train.coding_language == 'Ruby']
    source_pawn = train[train.coding_language == 'SourcePawn']
    # create list
    languages = [c_sharp, c , c_plus_plus, html, java, java_script, lua, php, python, ruby, source_pawn]
    # create figure
    plt.figure(figsize = (20, 11), edgecolor = 'darkslategrey')
    for i, language in enumerate(languages):    # loop through enumerated list
        plot_number = i + 1     # assign variable to increasing plot number
        plt.subplot(3, 4, plot_number)    # create subplot
        language['sentiment'].hist(color = 'indigo', edgecolor = 'black', bins = 20)    # plot histogram of feature
        plt.tight_layout()    # clear it up
        plt.xticks(rotation = 45, size = 11)    # rotate x-axis label ticks 45 degrees, increase size to 11
        plt.yticks(size = 13)    # increasee y-axis label ticks to size 13
        plt.xlabel('Sentiment Score')
        plt.ylabel('# of Occurrences')
        plt.grid(False)
        plt.title(f'Distribution of {language.coding_language.min()} Sentiment', size = 13) 
    return plt.show();

def unique_word_viz(train):
    '''
    This function returns a visualization of the Number of Words Unique to Each Language
    From Each Language's 20 most frequently occurring words.
    '''
    from prepare import basic_clean
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
    # create dataframes of each language's top 20 words
    javascript_top = pd.DataFrame(word_counts.sort_values('javascript', ascending = False).head(20))
    js_solo = javascript_top[(javascript_top.javascript == javascript_top['all'])]
    csharp_top = pd.DataFrame(word_counts.sort_values('c_sharp', ascending = False).head(20))
    cs_solo = csharp_top[(csharp_top.c_sharp == csharp_top['all'])]
    php_top = pd.DataFrame(word_counts.sort_values('php', ascending = False).head(20))
    php_solo = php_top[(php_top.php == php_top['all'])]
    c_top = pd.DataFrame(word_counts.sort_values('c', ascending = False).head(20))
    c_solo = c_top[(c_top.c == c_top['all'])]
    sourcepawn_top = pd.DataFrame(word_counts.sort_values('sourcepawn', ascending = False).head(20))
    sp_solo = sourcepawn_top[(sourcepawn_top.sourcepawn == sourcepawn_top['all'])]
    html_top = pd.DataFrame(word_counts.sort_values('html', ascending = False).head(20))
    html_solo = html_top[(html_top.html == html_top['all'])]
    c_plus_plus_top = pd.DataFrame(word_counts.sort_values('c_plus_plus', ascending = False).head(20))
    cp_solo = c_plus_plus_top[(c_plus_plus_top.c_plus_plus == c_plus_plus_top['all'])]
    java_top = pd.DataFrame(word_counts.sort_values('java', ascending = False).head(20))
    j_solo = java_top[(java_top.java == java_top['all'])]
    python_top = pd.DataFrame(word_counts.sort_values('python', ascending = False).head(20))
    py_solo = python_top[(python_top.python == python_top['all'])]
    lua_top = pd.DataFrame(word_counts.sort_values('lua', ascending = False).head(20))
    l_solo = lua_top[(lua_top.lua == lua_top['all'])]
    ruby_top = pd.DataFrame(word_counts.sort_values('ruby', ascending = False).head(20))
    r_solo = ruby_top[(ruby_top.ruby == ruby_top['all'])]
    # combine dataframes
    unique_words = pd.concat([js_solo, cs_solo, php_solo, c_solo, sp_solo, html_solo, cp_solo, j_solo, py_solo, l_solo, r_solo])
    # convert to binary
    unique_words[unique_words > 0] = 1
    # plot
    unique_words.drop(columns='all').sum().plot.bar().set(title='Number of Words Unique to Each Language\nFrom Each Language\'s Top 20 Words', xlabel='Language', ylabel='# of Words')
    plt.xticks(rotation=45)
    return plt.show();
