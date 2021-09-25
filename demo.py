#%% 
import pandas as pd
USER_DATA = './resources/data/users.json'
# %%
df = pd.read_json(USER_DATA, orient="index")
print(df.head(5))

# %%
print(df.columns)

# %%
