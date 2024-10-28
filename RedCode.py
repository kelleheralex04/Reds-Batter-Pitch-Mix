import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.inspection import permutation_importance
from sklearn.inspection import PartialDependenceDisplay
import seaborn as sns



# Step 1: Read in the dataset

data = pd.read_csv('RedsBatterData.csv')

# Step 2: Remove rows where 'PITCH_TYPE' or 'IF_FIELDING_ALIGNMENT' is empty
data = data.dropna(subset=['PITCH_TYPE', 'IF_FIELDING_ALIGNMENT'])

# Step 3: Replace NaN values in 'EVENTS' and 'BB_TYPE' columns with 'NA'
data['EVENTS'].fillna('NA', inplace=True)
data['BB_TYPE'].fillna('NA', inplace=True)
data = data.replace(pd.NA, 'NA')

# Step 4: Map pitch types to pitch categories
pitch_class_mapping = {
    'CH': 'OffSpeed', 'CS': 'Breaking', 'CU': 'Breaking', 'EP': 'Other',
    'FA': 'Other', 'FC': 'Fastball', 'FF': 'Fastball', 'FO': 'OffSpeed',
    'FS': 'OffSpeed', 'KC': 'Other', 'KN': 'Breaking', 'PO': 'Other',
    'SC': 'OffSpeed', 'SI': 'Fastball', 'SL': 'Breaking', 'ST': 'Breaking',
    'SV': 'Breaking'
}
data['Class'] = data['PITCH_TYPE'].map(pitch_class_mapping)

# Step 5: Remove rows where 'Class' is 'Other'
data = data[data['Class'] != 'Other']
# This was 2.53% of the data


# Step 6: Create a new column 'BALLS_STRIKES'
# This is to create a variable to represent the count
data['BALLS_STRIKES'] = data['BALLS'].astype(str) + "_" + data['STRIKES'].astype(str)

# Step 7: Group by BATTER_ID, GAME_YEAR, BALLS_STRIKES to calculate player stats
numeric_columns = ['ESTIMATED_WOBA_USING_SPEEDANGLE', 'DELTA_RUN_EXP']
player_stats = data.groupby(['BATTER_ID', 'GAME_YEAR', 'BALLS_STRIKES'])[numeric_columns].mean().reset_index()

# Step 8: Merge original categorical columns back into the aggregated dataset
categorical_columns = [ 'BAT_SIDE', 
                      
                       'IF_FIELDING_ALIGNMENT', 'OF_FIELDING_ALIGNMENT', 'TYPE']

# Perform left join to merge categorical columns
# Merge player_stats with the original data on common columns (like BATTER_ID and GAME_YEAR)
training_data = pd.merge(player_stats, data[['BATTER_ID', 'GAME_YEAR', 'BALLS_STRIKES',  
                                             'BAT_SIDE', 
                                               'IF_FIELDING_ALIGNMENT', 
                                             'OF_FIELDING_ALIGNMENT','TYPE']],
                         on=['BATTER_ID', 'GAME_YEAR', 'BALLS_STRIKES'])

# Step 9: Calculate pitch distribution (Breaking, Fastball, OffSpeed counts)
pitch_distribution = data.groupby(['BATTER_ID', 'GAME_YEAR', 'BALLS_STRIKES'])['Class'].value_counts().unstack(fill_value=0).reset_index()
pitch_distribution = pitch_distribution[['BATTER_ID', 'GAME_YEAR', 'BALLS_STRIKES', 'Breaking', 'Fastball', 'OffSpeed']]

# Merge pitch distribution into training_data
training_data = pd.merge(training_data, pitch_distribution, on=['BATTER_ID', 'GAME_YEAR', 'BALLS_STRIKES'])

# Step 10: Calculate pitch percentages
training_data['Total_Pitches'] = training_data['Breaking'] + training_data['Fastball'] + training_data['OffSpeed']
training_data['Breaking_pct'] = training_data['Breaking'] / training_data['Total_Pitches']
training_data['Fastball_pct'] = training_data['Fastball'] / training_data['Total_Pitches']
training_data['OffSpeed_pct'] = training_data['OffSpeed'] / training_data['Total_Pitches']

# Step 11: Shift pitch distribution percentages to predict next year's distribution
training_data['Next_Year'] = training_data['GAME_YEAR'] + 1
training_data['Breaking_pct_next'] = training_data.groupby('BATTER_ID')['Breaking_pct'].shift(-1)
training_data['Fastball_pct_next'] = training_data.groupby('BATTER_ID')['Fastball_pct'].shift(-1)
training_data['OffSpeed_pct_next'] = training_data.groupby('BATTER_ID')['OffSpeed_pct'].shift(-1)

# Drop rows with missing target values
training_data = training_data.dropna(subset=['Breaking_pct_next', 'Fastball_pct_next', 'OffSpeed_pct_next'])


# Step 12: One-hot encode categorical variables
training_data = pd.get_dummies(training_data, columns=categorical_columns, drop_first=True)

# Change rows without a ball hit into play to zero instead of Nan values
training_data['ESTIMATED_WOBA_USING_SPEEDANGLE'].fillna(0, inplace=True)

# Drop rows where 'DELTA_RUN_EXP' is NaN
# 4 rows
training_data = training_data.dropna(subset=['DELTA_RUN_EXP'])


# Check for null values in the entire training data
null_values = training_data.isnull().sum()

# Print columns with null values if 
#print(null_values[null_values > 0])

# Step 13: Define features (X) and target variables (y)
X = training_data.drop(['GAME_YEAR', 'Next_Year', 'Breaking_pct_next', 'Fastball_pct_next', 'OffSpeed_pct_next'], axis=1)
y = training_data[['Breaking_pct_next', 'Fastball_pct_next', 'OffSpeed_pct_next']]

# Step 14: Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=50)

# Step 15: Train a Random Forest Regrssion Model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Step 16: Make predictions
y_pred = rf_model.predict(X_test)

# Step 17: Evaluate the model using MSE, RMSE, R², and MAE
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

# Round and print evaluation metrics
print(f"MSE: {round(mse, 4)}")
print(f"RMSE: {round(rmse, 4)}")
print(f"R² Score: {round(r2, 4)}")
print(f"MAE: {round(mae, 4)}")

# Perform 5-fold cross-validation and print rounded average score
cv_scores = cross_val_score(rf_model, X, y, cv=5)
print(f'Average cross-validation score: {round(cv_scores.mean(), 4)}')

# Extract feature importances from the Random Forest model and display rounded values
feature_importances = rf_model.feature_importances_
importance_df = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=False)

# Round the 'Importance' values to three decimal places
importance_df['Importance'] = importance_df['Importance'].round(4)
print(importance_df)

# Plot the feature importances
plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'], importance_df['Importance'])
plt.xlabel('Importance')
plt.title('Feature Importances from Random Forest Regressor')
plt.show()


# ========================
# Make Predictions for 2024
# ========================

# Step 1: Read the new file with batter IDs for 2024 predictions
new_batter_data = pd.read_csv('RedsBatterPredictions.csv')

# Step 2: Filter the training data to include only rows from 2023 for each of the batters in the new file
prediction_data_2023 = training_data[(training_data['GAME_YEAR'] == 2023) & 
                                     (training_data['BATTER_ID'].isin(new_batter_data['BATTER_ID']))]


# Drop columns like 'GAME_YEAR' and any target columns used for training (optional based on model structure)
prediction_data_2023 = prediction_data_2023.drop(columns=['GAME_YEAR', 'Breaking_pct_next', 'Fastball_pct_next', 'OffSpeed_pct_next'], errors='ignore')

# Step 3: Ensure the column structure matches what the model expects
prediction_data_2023 = prediction_data_2023.reindex(columns=X.columns, fill_value=0)

# Step 4: Make predictions for 2024 using the trained model
predicted_2024 = rf_model.predict(prediction_data_2023)

# Step 5: Create a new DataFrame to store the predictions with the Total_Pitches column for weighting
predictions_2024_df = pd.DataFrame(predicted_2024, columns=['PITCH_TYPE_BB', 'PITCH_TYPE_FB', 'PITCH_TYPE_OS'])
predictions_2024_df['BATTER_ID'] = prediction_data_2023['BATTER_ID'].values
predictions_2024_df['Total_Pitches'] = prediction_data_2023['Total_Pitches'].values

# Step 6: Calculate weighted averages for each batter using 'Total_Pitches' as the weight
weighted_predictions = predictions_2024_df.groupby('BATTER_ID').apply(
    lambda group: pd.Series({
        'PITCH_TYPE_FB': (group['PITCH_TYPE_FB'] * group['Total_Pitches']).sum() / group['Total_Pitches'].sum(),
        'PITCH_TYPE_BB': (group['PITCH_TYPE_BB'] * group['Total_Pitches']).sum() / group['Total_Pitches'].sum(),
        'PITCH_TYPE_OS': (group['PITCH_TYPE_OS'] * group['Total_Pitches']).sum() / group['Total_Pitches'].sum(),
    })
).reset_index()

# Step 7: Merge with new_batter_data to get the PLAYER_NAME column
weighted_predictions = weighted_predictions.merge(new_batter_data[['BATTER_ID', 'PLAYER_NAME']], on='BATTER_ID', how='left')

# Step 8: Add GAME_YEAR column and arrange columns in the specified order
weighted_predictions['GAME_YEAR'] = 2024
weighted_predictions = weighted_predictions[['BATTER_ID', 'PLAYER_NAME', 'GAME_YEAR', 'PITCH_TYPE_FB', 'PITCH_TYPE_BB', 'PITCH_TYPE_OS']]


#Weight the predictions to match the format of the sample submission file
average_predictions = weighted_predictions[['PITCH_TYPE_FB', 'PITCH_TYPE_BB', 'PITCH_TYPE_OS']].mean()


# Assuming 'BATTER_ID' is a unique identifier in new_batter_data
missing_batters = new_batter_data[~new_batter_data['BATTER_ID'].isin(weighted_predictions['BATTER_ID'])]

# Step 9: Create a DataFrame to hold the average predictions for the missing batters
average_fill = pd.DataFrame({
    'BATTER_ID': missing_batters['BATTER_ID'],
    'PLAYER_NAME': missing_batters['PLAYER_NAME'],  # Ensure to keep PLAYER_NAME
    'GAME_YEAR': 2024,  # Add the year
    'PITCH_TYPE_FB': average_predictions['PITCH_TYPE_FB'],
    'PITCH_TYPE_BB': average_predictions['PITCH_TYPE_BB'],
    'PITCH_TYPE_OS': average_predictions['PITCH_TYPE_OS']
})

# Step 10: Combine the original predictions with the average fill DataFrame
# This is to create estimates for players with no 2023 data. 4/315 batters had no 2023 data
final_predictions = pd.concat([weighted_predictions, average_fill], ignore_index=True)

# Step 11: Sort the final DataFrame by BATTER_ID to maintain order
final_predictions = final_predictions.sort_values(by='BATTER_ID').reset_index(drop=True)

# Step 12: Save the final predictions to a new CSV file
final_predictions.to_csv('predictions.csv', index=False)

# Optional: Print the final DataFrame to verify
print(final_predictions.head())