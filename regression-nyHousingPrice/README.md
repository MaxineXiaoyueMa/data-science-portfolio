# supervised - regression - NY housing price
**Built a model to predict residential housing price within a New York county. Mean Absolute Error is expected to be $72,600, which is a $7,400 improvement from mean appraisal errors.**

*project notebooks rendered in nbviewer for better navigation: https://bit.ly/max-housingPrice-nbv*

*Disclaimer: the project scenario and data is from EliteDataScience's Machine Learning Master Class. Analysis is based on the curriculum with expansions of my own.*

## Project Objective
A NY based **REIT** (Real Estate Investment Trust) invests in residential housing units within its county. It has a regular business need of predicting housing price in order to evaluate investment opportunities.

Currently, they hire appraisers for estimation. This approach takes time and resource, and the appraiser's estimation is, on average, off by $80,000 per transaction.

Our goal for this project is **to build a model that can quickly provide a reasonable estimate that either matches or beats the accuracy of the appraisals**.

## Project Specifics
- **Delieverable**: trained final model
- **Machine Learning Task**: regression
- **Target Variable**: transaction price
- **Win Condition**: MAE < $80,000

## Data
- Historical housing transaction data in the county where the REIT resides in upstate New York (**1883 x 26**)
- Features include:
    - information on the neighborhood, e.g., number of grocery stores, bars, schools;
    - information on housing exteriors, e.g., lot size, roof material, wall material;
    - information on interiors, e.g., number of bathrooms, number of bedrooms, and whether there is a basement.

## Findings & Insights
1. Observations
     1. Insurance, property tax, number of rooms, house size positively correlate with transaction price, as expected.
     2. Apartments are smaller, thus cheaper. Apartments are more likely to locate in urban areas, therefore, there are more facilities like restaurants and gyms around. Evidently, neighborhood facility features are negatively correlated with price.
2. Insights
     1. Our final model can predict housing price with a **mean abslute error of $72,600**, which is $7,400 lower than appraisal estimate errors.
     2. Many factors have predictive power according to feature importance measures, no wonder it is hard for appraisers to make accurate estimates.
          1. As expected and observed, **insurance, property tax, property age** and  **size** are the most predictive features.
          2. **Recession indicator** also holds predictive power, which makes sense as macro environment highly influences real estate market. We could add more macro economic factors for further improvement, such as fed rate.
          3. **Neighborhood quality** has some predictive power, but to a lesser degree in testing data, possibly indicating where overfitting occured. Reason might be many features describing the same latent factor - convenience.
     3. To my surprise, **building materials**, however, be it wall or roof, are of almost no importance in predicting the price. Probably because materials these days are all of good quality.

## Further Improvements
1. Prediction could be more streamlined by automating data input and output.
2. Model could be further optimized.
3. More features could be introduced, such as:
    1.  fed rate as a **macro economic** indicator,
    2.  crime rate as a **neighborhood safty** measure,
    3.  number of public transit stops as a **neighborhood convenience** measure.

## File Structure

- **dev**: analysis notebooks
    - p1-EDA
    - p2-Data Cleaning + Feature Engineering
    - p3-Model Selection
        - Model candidates: **lasso/ridge/elastic-net/svr/random forest/extra tree/adaboost/greadientboosted trees**
        - Hyperparameter Tuning: **grid search, baysian optimization with *hyperopt***
    - p4-Model Evaluation + Model Delivery
- **deliverables**: final model for delievery

**Thank you for stopping by, feel free to reach out with anything you would like to share!**
