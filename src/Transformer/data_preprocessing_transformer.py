
# Import necessary libraries
import pandas as pd
import	numpy as np

# Read data file
df = pd.read_csv('sitges_access.csv')

# Drop unnecessary columns
df = df.drop(['Unnamed: 0', "server_name" , 'IP', 'logname', 'authenticate', 'date', 'referer'], axis=1)

# chnage bytes column name to bytess
df = df.rename(columns={'bytes': 'bytess'})

# change user-agent column name to user_agent
df = df.rename(columns={'user-agent': 'user_agent'})

# Change the level column to absolute values
df["level"] = df['level'].abs()

# Filter the DataFrame for value 0.000000 and other values
df_zero = df[df['level'] == 0]
df_non_zero = df[df['level'] != 0]

# Adjust the number of rows for 0.000000 to 200000
if len(df_zero) > 200000:
    df_zero = df_zero.sample(n=200000, random_state=1)  # Downsample
elif len(df_zero) < 200000:
    # Upsample (if necessary, can be adjusted based on actual needs)
    df_zero = df_zero.sample(n=200000, replace=True, random_state=1)

# Concatenate the adjusted DataFrame with the non-zero DataFrame
final_df = pd.concat([df_zero, df_non_zero])

# One-hot encode the petition column
final_df = pd.get_dummies(final_df, columns=['petition'])

# List of columns you want to ensure exist in the DataFrame
columns_to_ensure = ["petition_-", "petition_CONNECT", "petition_GET", "petition_HEAD", "petition_OPTIONS", "petition_POST", "petition_USER", "petition_PUT"]

# Loop through each column in the list
for column in columns_to_ensure:
    # If the column does not exist in the DataFrame, add it with default value 0
    if column not in final_df.columns:
        final_df[column] = 0

# Now that all columns exist and have default values if they were missing, convert them to integers
final_df[columns_to_ensure] = final_df[columns_to_ensure].astype(int)

# Convert boolean values in 'status_1' to integers (True to 1, False to 0)
final_df[["petition_-", "petition_CONNECT", "petition_GET", "petition_HEAD", "petition_OPTIONS", "petition_POST", "petition_USER", "petition_PUT"]] = final_df[["petition_-", "petition_CONNECT", "petition_GET", "petition_HEAD", "petition_OPTIONS", "petition_POST", "petition_USER", "petition_PUT"]].astype(int)

# copy final_df to df
df = final_df.copy()

# change petition_- column name to petition__
df = df.rename(columns={'petition_-': 'petition__'})

# Normalize 'level' column and overwrite the existing column
df['level'] = (df['level'] - df['level'].min()) / (df['level'].max() - df['level'].min())

# Invert the 'level' column
df["level"] = 1 - df['level']

# make train, validation and test data with 60%, 20% and 20% respectively, keep the same distribution of level column
train, validate, test = \
              np.split(df.sample(frac=1, random_state=1),
                       [int(.6*len(df)), int(.8*len(df))])

# save the train and test data to csv files
train.to_csv('train.csv', index=False)
validate.to_csv('validate.csv', index=False)
test.to_csv('test.csv', index=False)