import pickle
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from xgboost import XGBClassifier

# 1. Setup
output_file = 'model.bin'
max_depth = 6
learning_rate = 0.1

# 2. Data Preparation
print("Loading and prepping data...")
df = pd.read_csv('data/bank-full.csv', sep=';')
df.columns = df.columns.str.lower().str.replace(' ', '_')
for c in df.dtypes[df.dtypes == 'object'].index:
    df[c] = df[c].str.lower().str.replace(' ', '_')
df.y = (df.y == 'yes').astype(int)

# 3. Training on full data
print(f"Training final model (depth={max_depth})...")
dv = DictVectorizer(sparse=False)
train_dict = df.drop('y', axis=1).to_dict(orient='records')
X_train = dv.fit_transform(train_dict)
y_train = df.y.values

model = XGBClassifier(max_depth=max_depth, learning_rate=learning_rate, n_estimators=100, random_state=1)
model.fit(X_train, y_train)

# 4. Save the objects
print(f"Saving model to {output_file}...")
with open(output_file, 'wb') as f_out:
    pickle.dump((dv, model), f_out)

print("Training successful!")