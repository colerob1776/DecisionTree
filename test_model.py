import pandas as pd 
from data_cleaning import age_category
from pandas_decision_tree import PandasDecisionTree


df = pd.read_csv('./phpMYEkMl.csv')

df['cabin_category'] = df['cabin']
df['cabin_category'] = df['cabin_category'].apply(lambda x: x[0][0])

df['age_category'] = df['age']
df['age_category'] = df['age_category'].apply(age_category)

tree = PandasDecisionTree(df, ['pclass', 'sex', 'age_category', 'sibsp', 'parch', 'embarked', 'cabin_category'], 'survived')

train, test = tree.split_data(.85)

#intialize Decision Tree with train data
train_frame = PandasDecisionTree(train, ['pclass', 'sex', 'age_category', 'sibsp', 'parch', 'embarked', 'cabin_category'], 'survived')
#train the model
train_frame.train()

# Test the model
print(train_frame.test(test))