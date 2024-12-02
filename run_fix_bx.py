import pandas as pd

# Load the dataset
df = pd.read_csv('dataset_v2/BX/BX-Users.csv', sep=';', header=0, names=["User-ID","Location","Age"], encoding='ISO-8859-1')

# Rename the columns as specified
df.rename(columns={
    "User-ID": 'user_id:token',
    "Location": 'location:token',
    "Age": 'age:token',
}, inplace=True)

# Save the processed DataFrame to a new file named 'BX.inter'
df.to_csv('dataset_v2/BX/BX.user', sep='\t', index=False)
