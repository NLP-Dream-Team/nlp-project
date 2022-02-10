
# Predicting the Programming Language of a repository based on the text of the README file

###### Joshua Wheeler, Mason Sherbondy, Rajaram Gautam, and Sophia Stewart - Februray 11, 2022 

###### Submitted To: Codeup Data Science Team

# Project Summary
In this project, we tried to explore the repositories in the GitHub by searching for word "zombie" under the search section of explore tab of the GitHub. We web scraped 230 repos with repo name, programming language and readme contents. We work through various stages of data s

# Business Goals
- To build data set through Web Scraping with repository name (repo), programming language (language), and readme contents as columns of our dataframe.
- To determine what language is primarily used in a repository based only on the contents of that repository's README file.


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


# Data dictionary
|Index | Column Name | Description 
|---|---|---|
|0 |  repo                  | repo names that pops up while searching word zombie                                
|1 |  language              | name of programming language used on repo                                
|2 |  readme_contents       | text present on readme of the repo                       
                


# Project Specifications

### Plan:
- We decided on team that we will search for repos on GitHub that will have words zombie on it. We made a function to web scrap repo names , programmming language, and readme content of repo and get it in dataframe.

### Acquire


### Prepare


### Explore

### Model & Evaluate

# Reproduce My Project

- You will need your own env file with database credentials along with all the necessary files listed below to run my final project notebook.
- Read this README.md
- Download the aquire_new.py, prep_new.py, model_new.py final.ipynb files into your working directory
- Add your own env file to your directory. (user, password, host)
- Run the final_report.ipynb notebook

# Next Steps:





