# PulpitMobility_Forecasting 

1. Problem Understanding

Predict the amount of solar energy (ALLSKY_SFC_SW_DWN) reaching the ground at any given hour based on weather parameters. Real-World Context: In the renewable energy sector, "Grid Stability" is the primary challenge. Solar power is intermittent—clouds can cause a sudden drop in generation. Grid operators need accurate short-term forecasts to know exactly when to ramp up backup power (like gas turbines) to prevent blackouts. My model serves as this critical "Forecasting Engine."

2. Data Cleaning & Strategic Decisions
The raw sensor data contained significant noise, requiring bold engineering decisions:
Handling Missing Targets: The dataset used -999.0 as a placeholder for missing values. I chose to drop these rows entirely (~40% of the dataset) rather than imputing them.
Reasoning: Imputing the Target Variable (Ground Truth) introduces artificial bias. If I guess the solar irradiance for missing hours, the model learns my guess, not reality. Training on fewer, high-quality rows is better than training on abundant, fake data.
Feature Selection: I removed the ALLSKY_KT (Clearness Index) column.
Reasoning: This feature had a massive number of missing values. While "cloud clearness" is theoretically useful, in practice, a feature that is missing 70% of the time adds more noise than signal to the model.

3. Feature Engineering: The "Cyclic Time" Insight
Standard machine learning models treat time linearly. To a model, Hour 23 (11 PM) and Hour 0 (Midnight) appear far apart (distance = 23). In reality, they are adjacent.
The Solution: I transformed Hour and Month into Sine and Cosine components (hour_sin, hour_cos).
The Impact: This transformation mapped time onto a 2D circle. This allowed the model to mathematically understand the daily cycle of the sun—specifically, that the conditions at 11:59 PM are nearly identical to 12:01 AM, preserving the temporal continuity of the data.

4. Model Selection: Random Forest Regressor
I selected a Random Forest Regressor over Linear Regression or LSTM (Deep Learning) for this specific challenge:
Non-Linear Relationships: Solar irradiance has a complex, non-linear relationship with Temperature and Pressure. A linear model cannot easily capture the "bell curve" shape of daily solar intensity.
Robustness to Outliers: Sensor data is prone to spikes (e.g., a bird covering a sensor). Random Forests average the results of multiple decision trees, making them naturally resistant to these individual anomalies.
Efficiency:Random Forest offered the highest accuracy-to-training-time ratio, avoiding the extensive architecture tuning required for Neural Networks.

5. Results & Visual Analysis
Metric: I used Root Mean Squared Error (RMSE).
Justification: In energy forecasting, large errors are exponentially worse than small ones (predicting full sun during a storm causes a grid failure). RMSE penalizes these large errors more heavily than MAE.
Visual Validation: The "Actual vs. Predicted" plot demonstrates that the model successfully captures the diurnal (daily) cycle.
Success: The model correctly predicts near-zero irradiance at night.
Success: The peaks align well with the actual sensor readings, proving the model has learned the relationship between Solar Zenith Angle (SZA) and Irradiance.


# PulpitMobility_Classification

1. Problem Understanding
Classify sales transactions into their likely Payment Method (Cash, Credit, or Installment) based on transaction details like Price, Quantity, and Date. Real-World Context: For a car dealership, predicting how a customer will pay is crucial for financial planning. "Cash" deals provide immediate liquidity, while "Installments" require long-term accounts receivable management. A predictive model helps the finance team forecast their cash flow mix for the upcoming quarter.

3. Data Preprocessing & Feature Engineering
The raw sales data required specific transformations to extract behavioral signals:
Temporal Features: I converted Sale_Date into Month and Is_Weekend.
Reasoning: Consumer spending behavior often shifts on weekends or during specific seasons (e.g., tax return season), which might correlate with credit vs. cash usage.
Noise Reduction: I dropped high-cardinality columns like Salesperson (9,000+ unique names) and Sale_ID.
Reasoning: These identifiers are unique to specific transactions and do not generalize to new data. Keeping them would lead to severe overfitting (the model memorizing names instead of learning patterns).
Target Encoding: I used LabelEncoder to transform the categorical targets (Cash, Credit, Installment) into numerical values (0, 1, 2) for the Random Forest classifier.

4. Model Selection: Random Forest Classifier
I selected a Random Forest Classifier over simpler models like Logistic Regression:
Handling Non-Linearity: The relationship between Price and Payment Method is rarely linear. For example, very cheap cars might be Cash, mid-range might be Credit, and luxury cars might be Installment. A Linear Regression cannot easily capture these "thresholds," but a Decision-Tree-based ensemble like Random Forest excels at it.
Class Balance: The dataset was balanced (~33% for each class), so I did not need to apply SMOTE or aggressive class weighting, allowing the Random Forest to train on the natural distribution of data.

5. Results & Critical Analysis (The "Why")
Model Accuracy: 34.20%. Interpretation: In a balanced 3-class problem, a random guess yields ~33.3% accuracy. My model's performance (34%) indicates it is effectively performing at baseline.
Why did this happen? (Critical Thinking) Instead of tuning hyperparameters endlessly, I analyzed the data to find the root cause:
Feature Importance vs. Correlation: The Feature Importance plot (Figure 1) shows Sale_Price as the dominant feature. However, a post-hoc correlation analysis revealed the correlation coefficient between Sale_Price and Payment_Method is near 0.00.
Conclusion: The provided dataset appears to be synthetic or randomized. There is no mathematical relationship between the Price of the car and the Payment Method in this specific file.
Visual Proof: The Box Plot (Figure 2) confirms this. The price distributions for Cash, Credit, and Installment are identical. In a real-world scenario, we would expect "Installment" plans to skew towards higher price points.

6. Business Value & Alternative Insights
Despite the lack of predictive signal for classification, I extracted value through EDA (Exploratory Data Analysis):
Seasonality Analysis (Figure 3): I generated a seasonality graph to track sales volume by month. Even without a predictive classifier, this visualization provides actionable intelligence, allowing the dealership to identify peak sales months and optimize inventory ordering.
