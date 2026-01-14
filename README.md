# PulpitMobility Internship Challenge
Project Overview
This repository contains solutions for the PulpitMobility Internship Challenge, demonstrating data science skills across two distinct domains:
1.Solar Irradiance Forecasting: A time-series regression problem to predict solar energy generation.
2.Car Showroom Classification: A classification problem to predict customer payment methods.

# Project 1: Solar Irradiance Forecasting

Problem Understanding: 
The objective was to forecast All-Sky Surface Shortwave Downward Irradiance (ALLSKY_SFC_SW_DWN), which represents the solar energy available for power generation in Watts/$m^2$4.

Real-World Context: 
Accurate forecasting is critical for energy grid operators to balance supply and demand. By predicting solar intensity, operators can efficiently manage backup power sources (like gas turbines) during cloud cover or peak demand.

Approach and Methodology.
Data Cleaning:
Identified that -999.0 was used as a placeholder for missing values. I chose to drop these rows rather than impute them to ensure the model trained only on "ground truth" data, avoiding artificial bias.Removed the ALLSKY_KT column due to excessive missing values (data sparsity).

Feature Engineering (Cyclic Time):

Converted linear time features (Hour, Month) into Sine and Cosine components. This transformation maps time onto a 2D circle, allowing the model to understand that Hour 23 (11 PM) is adjacent to Hour 0 (Midnight), preserving temporal continuity.

Model Selection:

Selected Random Forest Regressor to handle the non-linear relationship between weather variables (Temperature, Pressure) and solar irradiance.

Hyperparameter Tuning:

Implemented RandomizedSearchCV with TimeSeriesSplit. This strict validation strategy ensures the model is evaluated on "future" data only, preventing data leakage found in standard k-fold cross-validation.

Tools and Technologies Used 

Language: Python 3.13
Libraries: pandas (Data Manipulation), numpy (Cyclic Transforms), scikit-learn (Modeling & Tuning), matplotlib/seaborn (Visualization).

Challenges Faced 

Data Quality: Approximately 40% of the target values were missing (-999.0). Deciding whether to impute or drop was a critical tradeoff. I prioritized data quality over quantity.

Design and Creative Decisions 

Metric Selection: I chose RMSE (Root Mean Squared Error) over MAE. In energy grid management, large errors (predicting sun during a storm) are exponentially more costly than small variances. RMSE penalizes these large errors heavily.

Visual Validation: I plotted the "Actual vs. Predicted" irradiance for the first 5 days of the test set. This visual confirmation proved the model successfully learned the diurnal (sunrise-sunset) cycle, providing confidence beyond just raw metrics.


# Project 2: Car Showroom Classification
Problem Understanding 

The goal was to classify sales transactions into their likely Payment Method (Cash, Credit, or Installment) based on features like Sale Price, Quantity, and Date.

Business Value: Predicting payment types helps dealerships forecast liquidity (Cash) versus accounts receivable (Installments), aiding in financial planning.

Approach and Methodology 

Preprocessing:

Derived temporal features: Month and Is_Weekend to capture seasonal or weekly spending behaviors.

Dropped high-cardinality noise: Removed Salesperson and Sale_ID to prevent the model from memorizing unique identifiers instead of learning patterns.

Model Selection:

Utilized a Random Forest Classifier to capture potential non-linear thresholds (e.g., high prices forcing installment plans) that linear models might miss.

Evaluation & Analysis:

Conducted a post-hoc Correlation Analysis and Feature Importance check to understand the drivers of the model's performance.

Tools and Technologies Used 

Language: Python 3.13

Libraries: pandas, scikit-learn (Random Forest, LabelEncoder), seaborn (Heatmaps, Boxplots).

Challenges Faced 

Low Predictive Signal: The initial model achieved an accuracy of ~34%, which is equivalent to random guessing for a 3-class problem.

Critical Resolution: Instead of aimlessly tuning the model, I investigated the data. I discovered that the correlation between Sale_Price and Payment_Method was near zero (-0.009). This proved the low accuracy was a data quality issue (synthetic/random data) rather than a modeling failure.

Design and Creative Decisions 

Pivot to Business Intelligence: Recognizing the model's limitations, I shifted focus from prediction to insight. I designed a Seasonality Graph to visualize sales volume trends by month.

Justification: Even if the predictive model is not production-ready, this visualization provides actionable value to the client (inventory planning for peak months), demonstrating a solution-oriented mindset.

Honest Reporting: I explicitly documented the low accuracy and provided statistical proof (Box Plots showing identical price distributions) to explain why it happened, rather than hiding the result. This reflects a commitment to analytical integrity.



