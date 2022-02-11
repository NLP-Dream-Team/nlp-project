
# Predicting the Programming Language of a repository based on the text of the README file

###### Joshua Wheeler, Mason Sherbondy, Rajaram Gautam, and Sophia Stewart - Februray 11, 2022 

###### Submitted To: Codeup Data Science Team

![](zombie.png)

# Project Summary
In this project, we tried to explore the repositories in the GitHub by searching for word "zombie" under the search section of explore tab of the GitHub. We web scraped 230 repos with repo name, programming language and readme contents. We work through various stages of data science pipeline acquire, prepare, explore, and modelling to ensure that data we prepare fits on the model where we feed.

# Business Goals
- To build data set through Web Scraping with repository name (repo), programming language (language), and readme contents as columns of our dataframe.
- To determine the programming language of repository based only on the contents of that repository's README file.
- To find the best model that predicts the programming language of the repo based on the contents of README file.


# Executive Summary
The goal of this project is to determine what language is primarily used in a repository based only on the contents of that repository's README file.

# Deliverables

- A well documented jupyter notebook report containing analysis.
- Live presentation of the work accomplished via zoom.
- Google Sides for general audience that summarizes findings.
- Github Repository with a complete readme.md, a final report(.ipynb), acquire, and prepare modules made to make workflow in project pipeline easy.

# Intial Questions
- What is the distribution of our Inverse Document Frequencies for the most common words?
- Does the length of the README vary by programming language?
- Do different programming languages use a different number of unique words?
- Does sentiment score vary by programming language?

# Data dictionary
|Index | Column Name | Description 
|---|---|---|
|0 |  repo                  | repo names that pops up while searching word zombie                                
|1 |  language              | name of programming language used on repo                                
|2 |  readme_contents       | text present on readme of the repo                       
|3 |  link                  | repo names that pops up while searching word zombie                                
|4 |  cleaned_readme        | clean readme after doing basic clean                               
|5 |  stemmed               | readme stemmed   
|6 |  lemmatized            | readme lemmatized                               
|4 |  message_length        | number of characters                               
|5 |  word_count            | number of words   




# Project Specifications

### Plan:
- We decided on group that we will search for repos on GitHub that will have words zombie on it. We made a function to web scrap repo names , programmming language, and readme content of repo and get it in dataframe. We plan to clean and prepare the readme contents by removing special characters, tokenize , stem, lemmatize, remove stopwords on string in the process of preparing it.

### Acquire
- Acquire was the most different and challanging part on this project, as we need to extract our data from github. Sometimes code work without problem and sometimes it throws us an error. So we saved extracted data on JSON and used pandas to make DataFrame out of it.

### Prepare
- Created a column with link to repository
- Cleaned readme contents using basic clean function defined in prepare module.
- Tokenized the clean readme contents
- Removed '\n' from the string
- Stemmed the words in the readme contents and made a column for it by removing stop words.
- Lemmatized the words in th ereadme contents and made a column for it by removing stop words. 
- Removed the records for programming language which gas less than 11 repos.
- Dropped the duplicates repos
- Dropped Nulls
- Removed repos with Non-English language as it is out of our scope of the research.


### Explore
We explored on data to get answer to four initials questions, we asked.
-Looking at the distribution of frequently occurring words was not helpful.Not all languages had most frequently occurring words that were unique to only that language, but most did.
-There was a significant amount of variation in the length of the README files based on what language they were primarily comprised of.
-There was a significant amount of variation in sentiment based on programming language.

### Model & Evaluate

###### Baseline
- Always predict JavaScript
- 12% Accuracy on Train
- 13% Accuracy on Validate

We used three models to evaluate the 
###### Logistic Regression
- Default hyperparameters
- 23% Accuracy on Train
- 20.45% Accuracy on Validate

###### Random Forest
- max_depth = 5
- 28% Accuracy on Train
- 13.64% Accuracy on Validate

###### Decision Tree 
- max_depth = 5
- 31% Accuracy on Train
- 9% Accuracy on Validate

###### Test
Based on the accuracy of train and validate, we can say that Random Forest and Decision Tree is overfitted since difference of accuracy and train is large. So we decided to use logistic regression to test our test data and gives us accuracy of 13.51% which beats baseline by 2.71%

- 13.51% Accuracy on Test
- Baseline 10.8% Accuracy on Test

# Reproduce My Project

- You will need your own env file with github_username and github_token along with all the necessary files listed below to run my final project notebook.
- Read this README.md
- Clone my repository in your local machine and run the report.ipynb
- Add your own env file to your directory. (github_username, github_token)
- Run the Report.ipynb notebook

# Next Steps:
With more time we would like to train our models using more repositories with a greater variety of subjects.