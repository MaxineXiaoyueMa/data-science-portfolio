# EndToEnd - Loan Default Prediction

Worked through project full lifecycle: **idea initiation, project scoping, data collection, project design, data cleaning, visualization, development, troubleshooting, and final delivery.**

Packages used: **Numpy, Pandas, Scikit-Learn, Matplotlib, Seaborn, Xgboost, Pickle, Datetime, Selenium**

-  **Reduced effective loan default rate from 20% to 6%.** 
-  **Scraped 2.6 million observations from lendingclub.com, each has 150 features.**
-  **Trained models remotely on AWS EC2 instances.**
-  **Packaged code into a custom model class, deployed it as an executable script**


<p>&nbsp;</p>

<a id = 'toc'></a>
**Table of contents**

0. [Overview](#over)
1. [Project Motivation](#motiv)
2. [Data](#data)
3. [Machine Learning Task and approach](#mltask)
4. [Findings and Result](#finding)
5. [Project File Structure](#file)
6. [Insights, surprises, lessons, further steps](#insight)

<p>&nbsp;</p>

<a id = 'over'></a>
## 0. Overview
- Machine learning task: **Supervised - Classification**
- Goal: **Identify loans with high default probability while keeping as many loans as possible to effectively reduce portfolio default rate.**
- Deliverables: **Trained model, python script for prediction, jupyter notebook for prediction**
- Candidate models: **Logistic Regression, SVM(Support Vector Machine), Naive Bayes, Random Forest, Gradient Boosted Trees(Xgboost)**
- Machine learning techniques: **Cross Validation, Grid Search, Preprocessing-Sandardization, Pipeline**
- Hardware: local and **AWS EC2**
- Software: **Jupyer notebook, Python**
- Packages: **Numpy, Pandas, Scikit-Learn, Matplotlib, Seaborn, Xgboost, Pickle, Datetime, Selenium**

**[back to top](#toc)**
<p>&nbsp;</p>


<a id = 'motiv'></a>
## 1. Project Motivation:
I support Peer-to-Peer(P2P) lending because I believe that:
1. everyone deserves the chance to get their finances together, and
2. we may all experience liquity crunch at times and could use a helping hand when we do, and
3. integrity is a virtue, but
4. integrity isn't always or solely reflected by our credit scores.

P2P lending helps to solve this problem by bypassing traditional middlemen such as banks or credit unions that tend to use credit scores as the sole measurement of credit worthiness. This way, it provides more access to capital for borrowers, and creates an alternative investment for investors.

It is rewarding to learn that many borrowers were able to secure loans that would otherwise not be availale to them to get out of debt or fulfill their dreams. Unfortunately, not all could keep their end of the bargain. The biggest risk for P2P lending is loan default. According to Lendingclub.com, the average interest rate for all issued loans is around 13%, but due to defaults, the realized return is only around 5%, a 60% decrease. Most importantly, it robbed capitals away from potential responsible borrowers.

So, **Can we build a model to predict a loan's probability of default, before issuance, so that investors can avoid unnecessary loss, and responsible borrowers will get the funding they need?** This project is set out to explore and accomplish this task.

**[back to top](#toc)**
<p>&nbsp;</p>

<a id = 'data'></a>
## 2. Data:
LendingClub.com, a leading P2P lending platform, publishes its loan data quaterly. All data used for this project is scraped from its website in December 2019 and contains **all loans issued from 2007 to the end of Q3 2019**.

Some facts about the data:
1. Number of observations: **over 2.6 million loans**, each loan ranges from $500 to $40,000;
2. **Target variable**: **'loan_status'** - only loans that are completely paid off or charged off are included for training as they are the only definite loan status to be relevant to our analysis.
3. Number of features: **150 features** that include information on loans, borrowers, borrowers' credit profile.

Feature examples:
1. Loan information:
    - `'term'`: duration of loan repayment, either 36 or 60 months
    - `'purpose'`: reasons for borrowing, e.g., debt consolidation, housing
2. Borrowers information:
    - `'emp_length'`: number of years the borrowerers have been employed
    - `'emp_title'`: borrowers' employment titles
3. Borrowers' credit information:
    - `'fico_range_high'`: borrowers' fico score range
    - `'mo_sin_old_il_acct'`: months since borrowerer opened their oldest installment accounts

**[back to top](#toc)**
<p>&nbsp;</p>

<a id = 'mltask'></a>
## 3. Machine learning approach:
1. The following classification models are trained, tuned, and compared:
    1. **Logistic Regression**: linear model, cheap to train compared to tree based models, traditionally good at loan default prediction tasks
    2. **Support Vector Machine**: similar to Logistic Regression, different loss function, interested in learning the difference in performance compared to logistic regression;
    3. **Naive Bayes**: cheap to train, strong assumptions on independence among features, traditionally good for NLP tasks, interested in learning the difference in performance compared to logistic regression;
    4. **Random Forest**: ensemble model - bagging, generally yields good result, but costly to train;
    5. **Xgboost**: ensemble model - boosting, great result, more expensive compared to linear models, worth investigation before cost and benefit analysis in model training.

2. **Train/test split**: The user case asks us to predict future returns on loans issued previousely, therefore, data is partitioned chronologically into three parts **(roughly 80/10/10)**. The first 80% of the data is used for hyperparameter tuning and model selection, the next 10% is used for threshold selection, and the last 10% is for model evaluation.

3. **Model Training, hyperparameter tuning, model selection**: use **grid search** method along with **time series cross validation** to compare model and hypermarater performance on **roc_auc** score.

4. **Threshold selection**:  **plot ROC curve** to visualize model performance and select thresholds with a **self defined metric** - `'effective default rate'`. `'effective default rate'` is calculated as the number of default loans the model fails to catch devided by the number of all remaining loans.

5. **[Project delivery](deliverables)**:
    1. A **[jupyter notebook](deliverables/predict_default.ipynb)** that requires only the final model and new data file location to display predictions inline.
    2. A **[python script](deliverables/predict_default.py)** that can be executed on the command line and saves predictions to a separate csv file.

**[back to top](#toc)**
<p>&nbsp;</p>

<a id = 'finding'></a>
## 4. Findings and result:
1. General observations:
    1. Out of the 2.6 million observations, about 540 thousand, or 20%,  are relevant to our training task.
    2. Loan default rate, for the entire portfolio, is 19.61%. However, the drop of return caused by the default is 60%, from 13.5% to 5% for the entire portfolio, an impact much bigger compared to its size.
    3. The number one reason borrowers take out loans is to consolidate debt, and number two is to pay off credit cards, both indicating borrowers seek out alternative access to capital in order to get their finances together.
    4. Default rate goes up as loan grade deteriorates as to be expected. However, interest rate does not go up proportinaly with default risk. E.g., A1 loans have a default rate of 3.3% and interest rate of 6.5%, D1 loans have a default rate of 27.5% and interest rate of 18.62%.
    5. Renters have a higher default rate (23%) compared to home owners (20%), but mortgage payers (17%) have the lowest default rate.
    6. Loans for small businesses have the highest default rate at 29% compared to all other loan purposes.

2. Interesting observations:
    1. Even at the lowest G5 grade, half of the loans are paid off.
    2. Top 20 earners, with annual income ranging from 4 million dollars to 9 million dollars, 4 defaulted, that's about 20% default rate. Most notebally, the borrower with the highest annual income makes over 9 million dollars a year. He/her borrowed $25,000 to pay off credit card and defaulted!
    3. Out of the top 10 loans with highest total collection amount against the borrowers, each ranging from 120K to 900K, only one defaulted.

I am fascinated by these observations, because they are, in a way, against our traditional judgements. It shows that integrity is an intricate trait that cannot be easily caputured by a few variables. The more we take into consideration, the better. Methods like machine learning has made it possible to examine relationships between hundreds and thousands of features at once. As a result, we now can and will continue to make better and more tailored decisions for each user with ease and on a mass scale. That is very exciting indeed!

3. Project results:
With our trained model and chosen thresholds, we can construct a portfolio:
    1. with an effective default rate of **6%**, reducing the defualt rate by **75%** by eliminating **98%** of all default loans, that is composed of **8%** of all loans, or
    2. with an effective default rate of **9%**, reducing the default rate by **60%** by eliminating **90%** of all default loans, that is composed of **25%** of all loans, or
    3. with an effective default rate of **11%**, reducing the default rate by **50%** by eliminating **80%** of all default loans, that is composed of **40%** of all loans.

The sweet spot personally is the second threshold, but one can make a decision based on their risk tolerance.

**[back to top](#toc)**
<p>&nbsp;</p>

<a id = 'file'></a>
## 5. Project file structure:

**readme.md**: project summary

**requirements.txt**: all packages needed to run the code

**[deliverables](deliverables)**: Final code in python and jupyer notebook to predict default probability for loans downloaded directly from lendingclub.com

**data**: data files and data dictionaries
  - raw: raw data files directly downloaded from lendinclub.com, too large to be included in git at this point
  - **abt**: data in numerical format ready for training, after data cleaning and feature engineering
  - **LCDataDictionary.xlsx**

**dev**: jupyter notebooks for analysis, important pickled variables, and notes from/for analysis

**preds**: new loan prediction utility functions and sample predictions

**[back to top](#toc)**
<p>&nbsp;</p>

<a id = 'insight'></a>
## 6. Lessons, improvements, and expansions:
### 1. Lessons:
1. When data cleaning: use unseen/new data's format, incl. data types and category values, to avoid inconsistencies with training data later, saving additional data cleaning steps. For example:
    1. 'term': numerical feature in unseen data vs. categorical in training data
    2. 'purpose' has slightly different category values, e.g., 'home_buying' in unseen data vs. 'house' in training data
2. When training on remote server, all the intermediate output file during training should be copied back to local, to avoid analysis inconsistence:
    - train_df.revol_util.median = 0.45 during imputing on local vs. 45 on remote due to modified data cleaning procedure on remote.
3. **Bug during webscraping**: A series of buttons need to be clicked in order to retrieve data files. I implemented an initial selenium workflow to automatically retrieve all files. However, during execution, several files were not selected. Since I couldn't figure out where the bug was, I ended up implementing a backup mechanism. The current set up caused some files to be downloaded twice. I posted [the question on Stackoverflow](#https://stackoverflow.com/questions/59168568/selenium-python-select-from-dropdown-click-button-modal-window-bug), but there is no answer as of 2020/03/09.

### 2. Improvements with more time and/or resources:
As an individual investor who only allocates a small portion of asset to P2P lending, having only 25% of all loans to choose from is more than enough to justify the 50% reduction in effective default rate. I may even go for keeping 8% of loans in order to  have a 6% default rate. However, I would consider the current project as a MVP(minimal viable product), and given more time and resources, the result can be improved.
1. **Model Tuning**:
    1. Logistic Regression hyperparameter C can be further tuned on a finer grid for potential improvement of performance. Although, the improvement may be limited, as the model's current performance on training set and validation set are already quite close.
    2. The model with the most potential for improvement is Random Forest. The current fitted model is overfitting, as indicated by its high training roc_auc of 0.94 and low validation score of 0.68. We could correct for overfitting by reducing min_samples_leaf, min_sample_split, and/or max_depth. Again, Random Forest is expensive to train.
2. **Features Selection**: NLP features and geospatial features that were dropped in data cleaning could be added back after appropriate processing
3. **Feature Engineering**: Macroeconomic indicators could be helpful, such as fed fund rate.
4. **Build a model for grade C&D only**: Identify the "underdogs" who has a harder time secure traditional types of fundings.
5. Build out structure using flask, AWS, etc., to automate the loan selection process.

### 3. Expansions with more time and/or resources:
1. Streamline the workflow further:
     1. automate loan data input,
     2. automate loan orders after prediction.

2. Borrower segmentation: Why do we borrow? Are there latent characteristics of borrowers?

3. Can we predict when borrowers need to borrow? How often they borrow? If so, assets can be allocated more dynamically to meet the demand better.

There is so much we could do! For the time being, however, this is enough to help me build my little loan portfolio. Moving on!

**[back to top](#toc)**


# Thank you for stopping by, I hope you find something interesting. Should you have any questions, insights, or anything you would like to share, feel free to reach out or submit a pull request!


**End of readme.md**

*Jupyter notebooks rendered in nbviewer [here](https://nbviewer.jupyter.org/github/MaxineXiaoyueMa/data-science-portfolio/tree/master/classification-loanDefault-webScrape-realData-AWS/dev/).*

**[back to top](#toc)**
