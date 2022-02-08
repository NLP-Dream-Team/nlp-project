import unicodedata
import re
import nltk
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer



def basic_clean(s):
    '''
    Takes a string and returns a normalized lowercase string 
    with special characters removed
    '''
    # lowercase
    s = str(s.lower())
    # normalize
    s = unicodedata.normalize('NFKD', s)\
    .encode('ascii', 'ignore')\
    .decode('utf-8', 'ignore')
    # remove special characters and lowercase
    s = re.sub(r"[^a-z0-9'\s]", '', s)
    return s

def tokenize(s):
    '''
    Takes a string and returns a tokenized version of the string
    '''
    # create tokenizer
    tokenizer = nltk.tokenize.ToktokTokenizer()
    # return tokenized string
    return tokenizer.tokenize(s, return_str=True)

def stem(s):
    '''
    Takes a string and returns a stemmed version of the string
    '''
    # create porter stemmer
    ps = nltk.porter.PorterStemmer()
    # apply stemmer
    stems = [ps.stem(word) for word in s.split()]
    # join list of words
    stemmed_s = ' '.join(stems)
    # return list of stemmed strings
    return stemmed_s

def lemmatize(s):
    '''
    Takes a string and returns a lemmatized version of the string
    '''
    # create lemmatizer
    wnl = nltk.stem.WordNetLemmatizer()
    # lemmatize split string
    lemmas = [wnl.lemmatize(word) for word in s.split()]
    # join words
    lemmatized_s = ' '.join(lemmas)
    # return lemmatized string
    return lemmatized_s

def remove_stopwords(s, extra_words = [], exclude_words = []):
    '''
    Takes a string and removes stopwords.
    Optional arguments: 
    extra_words adds words to stopword list
    exclude_words words to keep
    '''
    # create stopword list
    stopword_list = stopwords.words('english')
    # remove excluded words
    stopword_list = set(stopword_list) - set(exclude_words)
    # add extra words
    stopword_list = stopword_list.union(set(extra_words))

    #### old version
    # if len(extra_words) > 0:
    #     stopword_list.append(word for word in extra_words)
    # if len(exclude_words) > 0:
    #     stopword_list.remove(word for word in exclude_words)
    
    # split string into word list
    words = s.split()

    # add word to list if it's not in the stopword_list
    filtered_words = [w for w in words if w not in stopword_list]
    # join the filtered words into a string
    s_without_stopwords = ' '.join(filtered_words)
    # return list with removed stopwords
    return s_without_stopwords

def prep_string_data(df, column, extra_words=[], exclude_words=[]):
    '''
    Takes in a dataframe, original string column, with optional lists of words to
    add to and remove from the stopword_list. Returns a dataframe with the title,
    original column, and clean, stemmed, and lemmatized versions of the column.
    '''
    df['clean'] = df[column].apply(basic_clean).apply(tokenize).apply(remove_stopwords, extra_words=extra_words, exclude_words=exclude_words)
    
    df['stemmed'] = df['clean'].apply(tokenize).apply(stem)

    df['lemmatized'] = df['clean'].apply(tokenize).apply(lemmatize)

    
    return df[['title', column,'clean', 'stemmed', 'lemmatized']]

def prep_repo_data(df):
    '''
    Takes in a dataframe and returns a prepared dataframe.
    '''
    # create column with full link to github repo
    df['link'] = 'https://github.com/' + df.repo
    # create cleaned version of readme
    df['clean'] = [tokenize(basic_clean(readme)) for readme in df.readme_contents]
    # remove \n from cleaned readme
    df['clean'] = [re.sub('[\n]','', readme) for readme in df.clean]
    # stem readme
    df['stemmed'] = [remove_stopwords(stem(readme)) for readme in df.clean]
    # lemmatize readme
    df['lemmatized'] = [remove_stopwords(lemmatize(readme)) for readme in df.clean]
    
    # gathering languages with >= 11 repos
    languages = df.language.value_counts()[df.language.value_counts() >= 11].index.to_list()
    df = df[df.language.isin(languages)]
    
    # drop duplicates
    df.drop_duplicates(inplace=True)
    
    # return prepared dataframe
    df.dropna(inplace=True)
    return df

def split_data(df):
    '''
    Takes in a dataframe and returns train, validate, test subset dataframes. 
    '''
    tfidf = TfidfVectorizer()
    X = tfidf.fit_transform(df.lemmatized)
    y = df.language
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = .2, stratify = y, random_state = 222)
    X_train, X_validate, y_train, y_validate = train_test_split(X_train,y_train, test_size = .3, stratify = y_train, random_state = 222)
    return X, X_train, X_validate, X_test, y_train,y_validate, y_test