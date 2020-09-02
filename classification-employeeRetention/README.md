# supervised - classification - employee retention
**Built a classification model that can predict the probablities of employees leaving with a ROC-AUC score of 0.98. Ranking employees based on the probabilities coupled with the reference table of thresholds and expected percentage of coverage, HR can reach out to most likely leavers based on their budget and resource.**

*project rendered in nbviewer for better navigation: https://bit.ly/max-employeeRetention-nbv*

*Disclaimer: The project scenario and data is borrowed from Elite Data Science's Machine Learning Masterclass. Analysis is loosely based on the curriculum, but mostly my own.*

## Project Objective
**Retaining quality employees** is a priority of any employer. Not only because talents are a company's most valuable asset, but also because it saves companies' scarse resources on filling a position and retraining new employees. Granted, an employee's decision to leave is somtimes prompted by life events and cannot be changed. But many times, it is due to problems that, had the employer had the knowledge of, could be solved in time to keep the employee.

Our client is an IT comany's **HR deparment**. Their current method is to conduct an exit interview upon an employee leaving. This approach has several flaws, the major one being it is post employee leaving, therefore, anything that could have been done is already too late. Also, both  interviewers' ability and interviewees' willingness to cooperate affect how effective inerviews are.

Therefore, our goal for this project is **to build a model using past employee data to predict the probability of one leaving while they are still with the employer**. This approach is both more objective and proactive. HR professionals can allocate more resources on reaching out to the more likely leavers to understand their needs and wants, making it possible to resolve problems timely. They could also derive important insights as to what drives employees away and implement company wide policies that could address these issues.

## Project Specifics
- **Delieverables**: Executable model script + User guide
- **Machine Learning Task**: Classification
- **Target Variable**: Employed(0) vs. Left(1)
- **Win Condition**: Best possible model

## Data
- Data of size **14249 x 10**.
- Features include information such as how long one has been employeed, their working hours, performance ratings, satisfcation scores.
- Target feature is **status**.

## Findings and Insights
1. Observations:
     1. 23% employees end up leaving the company. On average, employees are involved in 3.7 projects simultaneously, and work 200 hours per month. Their average evaluation is at 0.7 and satisfaction is at 0.6. 14% of the employees have filed complaints, 2% received recent promotions. They have worked with the company for an average of 3 and a half years.
     2. **Leavers**: longer hour, less promotion, lower paid, most don't bother with filing complaints
         1. low or high monthly hours: under utilized and burn outs
         1. low or high evaluation: underperformers and overachievers
         1. mainly low satisfaction, but some highly satisfied leaves, possibly because they are looking for more challenging work, or life events.
         2. most leavers are entry level income earners, smaller fraction from middle income level, and very few from the top earners, which is to be expected
2. After modeling:
     1. Four factors have the most prediction power: **how many projects an employee is simultaneously working on, how many hours an employee works, how satisfied they are, and how long they have been with the company**, which is consistent with our observations above. Therefore, on the company level, HR and managers can work closely to monitor employee's workload, e.g., increasing responbilities for underutilized employees and reducing the load for employees who are buring out. Also, high **dissatisfaction** shouldn't be overlooked either.
     2. A probability of leaving is predicted for each employee. This number can best be used to **rank employees based on their likelihood of leaving, or compare two employees** rather than using it in the absolute manner.
     3. A **reference table** is generated from our model and training data to provide a guidance for our client to determine **how many employees to interview** in order to:
        1. reach certain percentage of employees who are going to leave, or
        2. reach certain number of employees who are going to leave,

        or, to determine **how many employees who are likely to leave we can reach** by:
        1. interviewing certain percentage of employees, or
        1. inverviewing certain number of employees.

        For example: to interview the top 10% of the employees ranked from the most likely to least likely would reach 37% of all possible leavers with 99% accuracy. (See user manual in delieverables for the example)
     4. When interviewing, an examination of the aforementioned four factors can provide some ideas to **drive the conversation** and discover whether an employee is having problems with their work, and if so, what. For example, an employee with a higher probability of leaving might be working on 6 projects and over 250 hours per month. It is highly possible that he/she is overworked and burnt out. (See user manual in delieverables for the example)

## Improvements
Our current model is quite accurate to address our client's problem. The current user interface could be improved to be more user friendly. Not every staff is comfortable with python, terminal or jupyter notebook. We can develop a dashboard that could:
1.  **highlight** employees who are at risk of leaving,
2.  **display** each employee's key feature values, and
3.  **track** actions taken by HR/Manager and corresponding result.

## File Structure
The project is structured in the following way:
- **dev**: analysis notebooks
    - p1-EDA
    - p2-Data Cleaning + Feature Engineering
    - p3-Model Selection + Model Evaluation:
        - Model candidates: logstic regression / random forest / gradient boosted trees
        - Parameter Tuning: grid search
    - p4-Model Delivery
- **delieverables**: delieverables for our client
- **model**: trained models

**Thank you for stopping by, feel free to reach out with anything you would like to share!**
