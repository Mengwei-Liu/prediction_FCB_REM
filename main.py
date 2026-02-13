import numpy as np
import pandas as pd
import math

# Function of getting season
def get_season(month):
    if 3<=month<=5:
        return 'Spring'
    if 6<=month<=8:
        return 'Summer'
    if 10<=month<=11:
        return 'Autumn'
    else:
        return 'Winter'

# Define the Rolling statistic function
def add_rolling_stats(df, team_prefix, window=3):
    # Extract the match data of two teams
    cols = [f"{team_prefix}_Yellow", f"{team_prefix}_Red", f"{team_prefix}_Goal"]

    # Compute the past N matches in order(exclude current raw)
    for col in cols:
        # mean
        df[f"{team_prefix}_Last{window}_Avg_{col.split('_')[-1]}"] = (
            df[col].shift(1)  # Exclude current raw
            .rolling(window=window, min_periods=1)
            .mean()
        )
    return df

# Function of Round off
def Round_off(value):
    decimal_part, integer_part = math.modf(value)
    if decimal_part <0.5:
        value=(int)(value)
    else:
        value=(int)(value+1)
    return value



# --- Load the historical dataset ---
df = pd.read_csv('dataset.csv')


# --- Preprocessing ---
# Preprocess the miss values
for col in ['FCB_Yellow','FCB_Red','REM_Yellow','REM_Red']:
    median_val = df[col].median()
    df[col].fillna(median_val)

# Convert the Column 'DAY' to the Date Form
df["DAY"] = pd.to_datetime(df["DAY"], errors="coerce")

# Extract the time feature
df['MONTH'] = df['DAY'].dt.month        # Extract the months
df['SEASON']= df['MONTH'].apply(get_season)       # Extract the seasons
df['WEEKDAY'] = df['DAY'].dt.weekday          # Extract the weekdays
df['IS_WEEKEND'] = df['WEEKDAY'].isin([5, 6]).astype(int)       #Annote the weekend

# One-Hot ecode the MONTH, SEASON and IS_WEEKEND
df = pd.get_dummies(df, columns=['MONTH', 'SEASON', 'WEEKDAY','IS_WEEKEND'])
# One-Hot ecode the match type
df = pd.get_dummies(df, columns=['TYPE'])

# Add the rolling statistics of the match data respectively
# Sort the matches by ascending time
df = df.sort_values('DAY').reset_index(drop=True)
# Handle with the data of two teams respectively
# For FC Barcelona
df = add_rolling_stats(df, team_prefix='FCB', window=3)
# For Real Madrid
df = add_rolling_stats(df, team_prefix='REM', window=3)
# Handle with lacked values
df.fillna(df[col].mean(), inplace=True)


# Standardized data
numeric_cols = ['FCB_Yellow', 'FCB_Red', 'FCB_Goal', 'REM_Yellow',
                'REM_Red', 'REM_Goal','FCB_Last3_Avg_Yellow', 'FCB_Last3_Avg_Red',
                'FCB_Last3_Avg_Goal', 'REM_Last3_Avg_Yellow', 'REM_Last3_Avg_Red',
                'REM_Last3_Avg_Goal'
                ]
# Save the value before standardizing
FCB_Yellow_mean=df['FCB_Yellow'].mean()
FCB_Yellow_std=df['FCB_Yellow'].std()
FCB_Red_mean=df['FCB_Red'].mean()
FCB_Red_std=df['FCB_Red'].std()
FCB_Goal_mean=df['FCB_Goal'].mean()
FCB_Goal_std=df['FCB_Goal'].std()
REM_Yellow_mean=df['REM_Yellow'].mean()
REM_Yellow_std=df['REM_Yellow'].std()
REM_Red_mean=df['REM_Red'].mean()
REM_Red_std=df['REM_Red'].std()
REM_Goal_mean=df['REM_Goal'].mean()
REM_Goal_std=df['REM_Goal'].std()
for col in numeric_cols:
    mean = df[col].mean()
    std = df[col].std()
# Standardization
for col in numeric_cols:
    df[col] = (df[col] - df[col].mean()) / df[col].std()
# Check if there is a Boolean column
bool_columns = df.select_dtypes(include=['bool']).columns
# Convert the Boolean column to an integer
for col in bool_columns:
    df[col] = df[col].astype(int)  # True → 1, False → 0

# Divide the dataset into training set and verified set
# Calculate the division points
total_samples = len(df)
train_size = int(total_samples * 0.8)  # The first 80% is the training set

# Divide the data set
train_df = df.iloc[:train_size]
verified_df = df.iloc[train_size:]

# Choose the feature and target
feature_columns = [
    'TYPE_CL', 'TYPE_FM', 'TYPE_IC', 'TYPE_KC','TYPE_LC',
    'TYPE_LL', 'TYPE_SC', 'FCB_Last3_Avg_Yellow', 'FCB_Last3_Avg_Red',
    'FCB_Last3_Avg_Goal', 'REM_Last3_Avg_Yellow', 'REM_Last3_Avg_Red','REM_Last3_Avg_Goal',
    'MONTH_1', 'MONTH_2', 'MONTH_3',
    'MONTH_4', 'MONTH_5', 'MONTH_6', 'MONTH_7', 'MONTH_8', 'MONTH_9',
    'MONTH_10', 'MONTH_11', 'MONTH_12', 'SEASON_Autumn', 'SEASON_Spring',
    'SEASON_Summer', 'SEASON_Winter', 'WEEKDAY_0', 'WEEKDAY_1', 'WEEKDAY_2',
    'WEEKDAY_3', 'WEEKDAY_5', 'WEEKDAY_6', 'IS_WEEKEND_0', 'IS_WEEKEND_1'
]
target_columns = [
    'FCB_Yellow', 'FCB_Red', 'FCB_Goal', 'REM_Yellow',
    'REM_Red', 'REM_Goal'
]

x_train = train_df[feature_columns]
y_train = train_df[target_columns]
x_test = verified_df[feature_columns]
y_test = verified_df[target_columns]
x_train = x_train.astype(np.float32)
x_test = x_test.astype(np.float32)
y_train = y_train.astype(np.float32)
y_test = y_test.astype(np.float32)

# --- Neural network ---
input_shape = x_train.shape[1]      # Input dimension
num_classes = y_train.shape[1]      # Output dimension

from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# Define the neural network
model = Sequential([
    Dense(256, activation='relu', input_shape=(input_shape,)),      # FC(Full connection) layer with 256 neurons
    Dropout(0.2),               # Randomly discard 20% of the neuron output to prevent overfitting of the model
    Dense(128, activation='relu'),           # Reduce to 128 neurons and learn more complex feature combinations
    Dropout(0.2),
    Dense(num_classes, activation='softmax')        # Output Layer
])

# Compiler the model
# Adam optimizer, automatically adjust the weights of the neural network during the training process
# Categorical_crossentropy is used to measure the difference between the predicted probability distribution of the model and the true label
# Evaluation index
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# Define callback function
callbacks = [
    ModelCheckpoint(
        filepath='best_model_keras.h5',     # Save path
        monitor='val_loss',                 # Monitor the loss of the validation set
        save_best_only=True,                # Only save the optimal model
        verbose=1                           # Display the saved information

    ),
    EarlyStopping(
        monitor='val_loss',                 # Monitor the loss of the validation set
        patience=10,                        # Admit there has been no improvement for 10 consecutive epochs
        restore_best_weights=True           # Recover the optimal weight
    )
]

print('------------------------------------------------------------------------------------------------------------------')
print('Start Training...')
# Training
history = model.fit(
    x_train, y_train,
    epochs=100,                             # The maximum training round
    batch_size=32,                          # The number of samples per batch
    validation_data=(x_test, y_test),       # Validate the data
    callbacks=callbacks,                    # Use the callback function
    verbose=1                               # Show process bar of training
)

# Evaluate the test set
# Load the optimal model
from tensorflow.keras.models import load_model
best_model = load_model('best_model_keras.h5')

# Evaluating
test_loss, test_acc = best_model.evaluate(x_test, y_test, verbose=0)
print(f'Test Accuracy: {test_acc:.4f}')
# Evaluate the confidence level
y_probs = best_model.predict(x_train)[0]
confidence = np.max(y_probs)
print(f"confidence level: {confidence:.2f}")
print('End Training...')
print('------------------------------------------------------------------------------------------------------------------')



import matplotlib.pyplot as plt

# ---Visual training process---
def plot_training_history(history):
    # Draw the loss curve
    plt.figure(figsize=(12, 5))

    # Loss curve
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss During Training')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Accuracy curve
    if 'accuracy' in history.history:
        plt.subplot(1, 2, 2)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy During Training')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

    plt.tight_layout()
    plt.show()

# Call the drawing function
plot_training_history(history)



# --- Predict the future data ---
# Load the dataset that we will predict
pf = pd.read_csv('prediction_data_set.csv')
# Dataset preprocess
# Convert the Column 'DAY' to the Date Form
pf["DAY"] = pd.to_datetime(pf["DAY"], errors="coerce")
for col in ['FCB_Yellow','FCB_Red','REM_Yellow','REM_Red']:
    if col in pf.columns:
        median_val = pf[col].median()
        pf.loc[pf['DAY'] <= '2025-04-26', col] = pf.loc[pf['DAY'] <= '2025-04-26', col].fillna(median_val)
# Extract the time feature
pf['MONTH'] = pf['DAY'].dt.month
pf['SEASON'] = pf['MONTH'].apply(get_season)
pf['WEEKDAY'] = pf['DAY'].dt.weekday
pf['IS_WEEKEND'] = pf['WEEKDAY'].isin([5, 6]).astype(int)


# One-Hot ecode the MONTH, SEASON and IS_WEEKEND
pf = pd.get_dummies(pf, columns=['MONTH', 'SEASON', 'WEEKDAY', 'IS_WEEKEND'])
# One-Hot ecode the match type
pf = pd.get_dummies(pf, columns=['TYPE'])
# Add the rolling statistics of the match data respectively
# Sort the matches by ascending time
pf = pf.sort_values('DAY').reset_index(drop=True)
# Handle with the data of two teams respectively
# For FC Barcelona
pf = add_rolling_stats(pf, team_prefix='FCB', window=3)
# For Real Madrid
pf = add_rolling_stats(pf, team_prefix='REM', window=3)
# Handle with lacked values
pf.fillna(df[col].mean(), inplace=True)

# Standardized data
numeric_cols = ['FCB_Yellow', 'FCB_Red', 'FCB_Goal', 'REM_Yellow',
                'REM_Red', 'REM_Goal','FCB_Last3_Avg_Yellow', 'FCB_Last3_Avg_Red',
                'FCB_Last3_Avg_Goal', 'REM_Last3_Avg_Yellow', 'REM_Last3_Avg_Red',
                'REM_Last3_Avg_Goal'
                ]
for col in numeric_cols:
    pf[col] = (pf[col] - mean) / std

# Check if there is a Boolean column
bool_columns = pf.select_dtypes(include=['bool']).columns

# Convert the Boolean column to an integer
for col in bool_columns:
    pf[col] = pf[col].astype(int)  # True → 1, False → 0

# Collect the input feature
x_pre=pf.iloc[-1][feature_columns]
x_pre = x_pre.values.reshape(1, -1).astype('float32')
# Predict
y_pre=best_model(x_pre)
#print(y_pre)       #test
y_pre=y_pre * std + mean
#print(type(y_pre))       # Tensor types in TensorFlow
y_list = y_pre.numpy().tolist()
# Extract the predicted data
B_yellow = y_list[0][0]
B_red = y_list[0][1]
B_goal = y_list[0][2]
R_yellow = y_list[0][3]
R_red = y_list[0][4]
R_goal = y_list[0][5]


# Data process
# Recovery data
B_yellow = B_yellow * FCB_Yellow_std + FCB_Yellow_mean
B_red = B_red * FCB_Red_std + FCB_Red_mean
B_goal = B_goal * FCB_Goal_std + FCB_Goal_mean
R_yellow = R_yellow * REM_Yellow_std + REM_Yellow_mean
R_red = R_red * REM_Red_std + REM_Red_mean
R_goal = R_goal * REM_Goal_std + REM_Goal_mean
# For the predicted data, use round off function
B_yellow = Round_off(B_yellow)
B_red = Round_off(B_red)
B_goal = Round_off(B_goal)
R_yellow = Round_off(R_yellow)
R_red = Round_off(R_red)
R_goal = Round_off(R_goal)
# Output the result
print('-----------------------------------------------')
print('Start Prediction...')
print(f'Yellow card of FC Barcelona is {B_yellow}')
print(f'Red card of FC Barcelona is {B_red}')
print(f'Goal of FC Barcelona is {B_goal}')
print(f'Yellow card of Real Madrid is {R_yellow}')
print(f'Red card of Real Madrid is {R_red}')
print(f'Goal of Real Madrid is {R_goal}')
print('-----------------------------------------------')
# Judge the winner
if B_goal > R_goal:
    print('Winner is FC Barcelona')
elif B_goal < R_goal:
    print('Winner is Real Madrid')
else:
    print('The result is Draw')

print('End Prediction...')
print('-----------------------------------------------')



















