import numpy as np
import pandas as pd
from sklearn import cross_validation
from sklearn import neighbors
from sklearn import grid_search
from sklearn import metrics
from sklearn import linear_model
from sklearn import dummy
from sklearn import ensemble

import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="whitegrid", font_scale=1)
# %matplotlib inline


# Load data
df = pd.read_csv("train.csv")

# Check info
print df.info()

# Check head
print df.head()

# Check outcome counts
print df.groupby('OutcomeType').count()


### PREPARE FOR DATA ANALYSIS ###

# Check counts for missing values in each column
print df.isnull().sum()

# Check data types
print df.dtypes

def wrangler(df):
    # Drop generic columns
    df = df.drop(['AnimalID', 'Name', 'DateTime'], axis=1)

    ### THERE IS STILL NON-NUMERIC VALUES IN THESE TWO COLUMNS ###
    ### MAKE SURE YOU FIX THIS ISSUE AND ADD THESE COLUMNS BACK IN ###
    # Drop generic columns
    df = df.drop(['SexuponOutcome'], axis=1)

    # Drop columns with too many missing values
    df = df.drop(['OutcomeSubtype'], axis=1)

    # Replace missing values using median value
    df.loc[(df['AgeuponOutcome'].isnull()), 'AgeuponOutcome'] = df['AgeuponOutcome'].dropna()

    return df

df = wrangler(df)

### EXPLORATORY DATA ANALYSIS ###

# Get summary statistics for data
df.describe()

# Get pair plot for data
# sns.pairplot(df)



### PREPARE FOR DATA MODELING ###


def preproc(df):
    df['AgeuponOutcome'] = df['AgeuponOutcome'].apply(lambda x: str(x).split()[0])
    di = {'nan': 0}
    df = df.replace({'AgeuponOutcome': di})
    df['AgeuponOutcome'] = df['AgeuponOutcome'].astype(int)

    ### TURN THESE BACK ON TO FIX THE NON-NUMERIC ISSUE ###
    di = {'Cat': 0, 'Dog': 1}
    df = df.replace({'AnimalType': di})

    # di = {'Intact Female': 0, 'Intact Male': 1, 'Neutered Male': 2, 'Spayed Female': 3, 'Unknown': 4}
    # df = df.replace({'SexuponOutcome': di})

    # Get dummy variables for Breed
    df = df.join(pd.get_dummies(df['Breed'], prefix='Breed'))
    # Remove Breed column
    df = df.drop(['Breed'], axis=1)

    # Get dummy variables for Color
    df = df.join(pd.get_dummies(df['Color'], prefix='Color'))
    # Remove Color column
    df = df.drop(['Color'], axis=1)

    return df

df = preproc(df)



### DATA MODELING ###

# Build a model to predict whether an animal was adopoted (1) or not (0)
df['y1'] = df['OutcomeType'].apply(lambda x: 1 if x=='Adoption' else 0)
df = df.drop(['OutcomeType'], axis=1)


print df.dtypes


# Set target variable name
target = 'y1'

# Set X and y
X = df.drop([target], axis=1)
y = df[target]

# Create separate training and test sets with 60/40 train/test split
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.4, random_state=42)

# Instantiate model
clf = ensemble.RandomForestClassifier(n_estimators=200)

# Train model on training set
clf.fit(X_train, y_train)


# Evaluate accuracy of model on test set
print "Accuracy: %0.3f" % clf.score(X_test, y_test)

# Evaluate ROC AUC score of model on test set
print 'ROC AUC: %0.3f' % metrics.roc_auc_score(y_test, clf.predict_proba(X_test)[:,1])


# Plot feature importances
feature_names = X.columns
feature_importances = clf.feature_importances_
feature_dict = dict(zip(feature_names, feature_importances))

features_df = pd.DataFrame(feature_dict.items(), columns=['Features', 'Importance Score'])
features_df.sort_values('Importance Score', inplace=True, ascending=False)
sns.barplot(y='Features', x='Importance Score', data=features_df)

print features_df.head(10)
