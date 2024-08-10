import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, brier_score_loss, precision_score, recall_score, f1_score, roc_auc_score, accuracy_score
from scipy.stats import ttest_ind
from collections import Counter
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Load data
db = pd.read_csv('final.csv')

db.head()
db = db.drop(['Unnamed: 16'],axis=1)
db['Covered_Recipient_First_Name'].fillna('unknown', inplace=True)
db['Covered_Recipient_Last_Name'].fillna('unknown', inplace=True)
db = db.set_index('Physician_NPI')
db.head()
group_cols = ['Physician_NPI']

agg_dict = {'Tot_Drug_Cst':['sum','mean','max'], \
           'Tot_Clms':['sum','mean','max'],\
           'Tot_Day_Suply':['sum','mean','max']}


db1 = db.groupby(group_cols).agg(agg_dict).astype(float)
db1.head()
if isinstance(db.index, pd.MultiIndex):
    db = db.reset_index()

if isinstance(db1.index, pd.MultiIndex):
    db1 = db1.reset_index()
 # db.reset_index(inplace=True)
# db1.reset_index(inplace=True)

# database = pd.merge(db, db1, how='left', on='Physician_NPI')


# Check the index levels
print(db.index.nlevels)  # Should be 1
print(db1.index.nlevels)  # Should be 1

# Check the column levels
print(db.columns.nlevels)  # Should be 1
print(db1.columns.nlevels)  # Should be 2

# Flatten the columns if necessary
if db1.columns.nlevels > 1:
    db1.columns = ['_'.join(col).strip() for col in db1.columns.values]

# Rename the column to match
db1.rename(columns={'Physician_NPI_': 'Physician_NPI'}, inplace=True)

# Verify columns
print(db.columns)
print(db1.columns)

# Perform the merge
database = pd.merge(db, db1, how='left', on='Physician_NPI')
fraud = db.loc[:,['NPI','EXCLTYPE']]
fraud.head()
fraud = fraud.query('NPI !=0')
fraud.count()
rename_dict = {'EXCLTYPE':'is_fraud'}
fraud = fraud.rename(columns=rename_dict)
fraud.head()                                                                                                                                    
fraud['is_fraud'] = 1
fraud.head()
data = pd.merge(database,fraud, how ='left',on = 'NPI')
data.head()
data['is_fraud'] = data['is_fraud'].fillna(0)
db.fillna(0, inplace=True)
db.head()
data[data['is_fraud']==1].count()
data_features=data
import numpy as np
data_features['Tot_Drug_Cst_sum_sum'] = data_features['Tot_Drug_Cst_sum'].map(lambda x: np.log10(x + 1.0))
data_features['Tot_Clms_sum_sum'] = data_features['Tot_Clms_sum'].map(lambda x: np.log10(x + 1.0))
data_features['Tot_Day_Suply_sum_sum'] = data_features['Tot_Day_Suply_sum'].map(lambda x: np.log10(x + 1.0))
data_features['Total_Amount_of_Payment_USDollars'] = data_features['Total_Amount_of_Payment_USDollars'].map(lambda x: np.log10(x + 1.0))

data_features['Tot_Drug_Cst_mean_mean'] = data_features['Tot_Drug_Cst_mean'].map(lambda x: np.log10(x + 1.0))
data_features['Tot_Clms_mean_mean'] = data_features['Tot_Clms_mean'].map(lambda x: np.log10(x + 1.0))
data_features['Tot_Day_Suply_mean_mean'] = data_features['Tot_Day_Suply_mean'].map(lambda x: np.log10(x + 1.0))

data_features['Tot_Drug_Cst_max_max'] = data_features['Tot_Drug_Cst_max'].map(lambda x: np.log10(x + 1.0))
data_features['Tot_Clms_max_max'] = data_features['Tot_Clms_max'].map(lambda x: np.log10(x + 1.0))
data_features['Tot_Day_Suply_max_max'] = data_features['Tot_Day_Suply_max'].map(lambda x: np.log10(x + 1.0))


data_features['claim_max-mean'] = data_features['Tot_Clms_max'] - data_features['Tot_Clms_mean_mean']

data_features['supply_max-mean'] = data_features['Tot_Day_Suply_max'] - data_features['Tot_Day_Suply_mean_mean']

data_features['drug_max-mean'] = data_features['Tot_Drug_Cst_max'] - data_features['Tot_Drug_Cst_mean_mean']
categorical_features = ['NPI','Covered_Recipient_Last_Name','Covered_Recipient_First_Name','Recipient_City', 'Recipient_State']
numerical_features = ['Tot_Drug_Cst_sum_sum', 'Tot_Drug_Cst_mean_mean','Total_Amount_of_Payment_USDollars',
       'Tot_Drug_Cst_max_max', 'Tot_Clms_sum_sum',
       'Tot_Clms_mean_mean', 'Tot_Clms_max_max',
       'Tot_Day_Suply_sum_sum', 'Tot_Day_Suply_mean_mean', 'Tot_Day_Suply_max_max',
    'claim_max-mean','supply_max-mean', 'drug_max-mean']
target = ['is_fraud']
allvars = categorical_features + numerical_features + target                                                     
y = data_features["is_fraud"].values
X = data_features[allvars].drop('is_fraud',axis=1)
# scikit learn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
#from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=0)
print(X_train.shape)
print(X_valid.shape)
X_train[numerical_features] = X_train.loc[:,numerical_features].fillna(0)
X_valid[numerical_features] = X_valid.loc[:,numerical_features].fillna(0)
X_train[categorical_features] = X_train.loc[:,categorical_features].fillna('NA')
X_valid[categorical_features] = X_valid.loc[:,categorical_features].fillna('NA')
scaler= StandardScaler()
X_train[numerical_features] = scaler.fit_transform(X_train[numerical_features].values)
X_valid[numerical_features] = scaler.transform(X_valid[numerical_features].values)
ix_ran = data_features.index.values
np.random.shuffle(ix_ran)

df_len = len(data_features)
train_len = int(df_len * 0.8)  # 80% for training


ix_train = ix_ran[:train_len]
ix_valid = ix_ran[train_len:]

# Use .iloc for integer-location based indexing
df_train = data_features.iloc[ix_train]
# Use .iloc for integer-location based indexing
df_valid = data_features.iloc[ix_valid]

print(len(ix_train))
print(len(ix_valid))
# Drug Weighted_Scores

partD_drug_train = pd.merge(database,df_train[['NPI','is_fraud']], how='inner', on=['NPI'])
partD_drug_All = pd.merge(database,data_features[['NPI','is_fraud']], how='inner', on=['NPI'])
print(len(partD_drug_train[partD_drug_train['is_fraud']==1]))
print("Total records in train set : ")
print(len(partD_drug_train))
print("Total Fraud in train set : ")
print(len(partD_drug_train[partD_drug_train['is_fraud']==1]))
partD_drug_train.head(50)
cols = ['Tot_Drug_Cst','Tot_Clms','Tot_Day_Suply']
partD_drug_train_Group = partD_drug_train.groupby(['DrugName', 'is_fraud'])
partD_drug_All_Group = partD_drug_All.groupby(['DrugName', 'is_fraud'])
drug_keys = partD_drug_train_Group.groups.keys()
print(len(drug_keys))
# # Create a list of unique drug names from your DataFrame
# get unique drug names
drugs = set([ drugx for drugx in partD_drug_train['DrugName'].values if isinstance(drugx, str)])
print(len(drugs))

drug_with_isfraud = [drugx for drugx in drugs if ((drugx,0.0) in drug_keys ) & ( (drugx,1.0) in drug_keys)]
print(drug_with_isfraud)
from scipy.stats import ttest_ind
re_drug_tt = dict()
for drugx in drug_with_isfraud:
    for colx in cols:
        fraud_0 = partD_drug_train_Group.get_group((drugx,0.0))[colx].values
        fraud_1 = partD_drug_train_Group.get_group((drugx,1.0))[colx].values
        # print len(fraud_0), len(fraud_1)
        if (len(fraud_0)>2) & (len(fraud_1)>2) :
            tt = ttest_ind(fraud_0, fraud_1)
            re_drug_tt[(drugx, colx)] = tt
#Setting Probilities
Prob_005 = [(key, p) for (key, (t, p)) in re_drug_tt.items() if p <=0.05]
print(len(Prob_005))
inx = 10
if inx < len(Prob_005):
    drug_name = Prob_005[inx][0][0]
    print(drug_name)
    df_bar = pd.concat([partD_drug_All_Group.get_group((Prob_005[inx][0][0], 0.0)),
                        partD_drug_All_Group.get_group((Prob_005[inx][0][0], 1.0))])
    df_bar.head()
else:
    print(f"Index {inx} is out of range for Prob_005 with length {len(Prob_005)}")
inx=10
drug_name = Prob_005[inx][0][0]
print(drug_name)
df_bar = pd.concat([partD_drug_All_Group.get_group((Prob_005[inx][0][0],0.0)), partD_drug_All_Group.get_group((Prob_005[inx][0][0],1.0))])
df_bar.head()
Feture_DrugWeighted = []
new_col_all =[]
for i, p005x in enumerate(Prob_005):
    #if i>4:
    #   break
    drug_name = p005x[0][0]
    cat_name = p005x[0][1]

    new_col = drug_name+'_'+cat_name
    new_col_all.append(new_col)

    drug_0 = partD_drug_All_Group.get_group((drug_name,0.0))[['NPI', cat_name]]
    drug_1 = partD_drug_All_Group.get_group((drug_name,1.0))[['NPI', cat_name]]

    drug_01 = pd.concat([drug_0, drug_1])
    drug_01.rename(columns={cat_name: new_col}, inplace=True)
    Feture_DrugWeighted.append(drug_01)
npi_col = data_features[['NPI']]

w_npi = []

for n, nx in enumerate(Feture_DrugWeighted):
      nggx = pd.merge(npi_col, nx.drop_duplicates(['NPI']), on='NPI', how='left')
      w_npi.append(nggx)
data_features1 = data_features
for wx in w_npi:
    col_n = wx.columns[1]
    data_features1[col_n] = wx[col_n].values

wx = w_npi[0]
wx.columns[1]
col_n = wx.columns[1]
len(wx[col_n].values)
data_features1.fillna(0)
# data_features1()
new_col_all
data_features1['drug_mean'] = data_features1[new_col_all].mean(axis=1)
data_features1['drug_mean'] = data_features1['drug_mean'].map(lambda x: np.log10(x + 1.0))
data_features1['drug_sum'] = data_features1[new_col_all].sum(axis=1)
data_features['drug_sum'] = data_features['drug_sum'].map(lambda x: np.log10(x + 1.0))
data_features1['drug_variance'] = data_features1[new_col_all].var(axis=1)
df_train = data_features1.iloc[ix_train]
df_valid = data_features1.iloc[ix_valid]

# Replace NaN values with 0 in both df_train and df_valid
df_train = df_train.fillna(0)
df_valid = df_valid.fillna(0)

# print(df_train)
# print(df_valid)
#Create the Specialty Weight
spec_dict =[]
spec_fraud_1 = df_train[df_train['is_fraud']==1]['SPECIALTY']
from collections import Counter
counts = Counter(spec_fraud_1)
spec_dict =  dict(counts)
data_features1['Spec_Weight'] = data_features1['SPECIALTY'].map(lambda x: spec_dict.get(x, 0))
df_train = data_features1.iloc[ix_train]
df_valid = data_features1.iloc[ix_valid]

# Replace NaN values with 0 in both df_train and df_valid
df_train = df_train.fillna(0)
df_valid = df_valid.fillna(0)
df_train.fillna(0)
len(df_train[df_train['is_fraud'] == 1])
numerical_features1 = numerical_features + ['drug_sum','Spec_Weight']
positives=len(df_train[df_train['is_fraud'] == 1])
positives
dataset_size=len(df_train)
dataset_size
per_ones=(float(positives)/float(dataset_size))*100
per_ones
negatives=float(dataset_size-positives)
t=negatives/positives
t
BalancingRatio= positives/dataset_size
BalancingRatio
BalancingRatio= positives/dataset_size
BalancingRatio
X= df_train[numerical_features1].values
Y = df_train['is_fraud'].values
clf =  LogisticRegression(C=1e5, class_weight={0:1, 1:4000}, n_jobs=3)
clf.fit(X,Y)
y_p=clf.predict_proba(X)
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn import ensemble
from sklearn.metrics import confusion_matrix, brier_score_loss, precision_score, recall_score, f1_score, roc_auc_score, accuracy_score
params_0 = {'n_estimators': 50, 'max_depth': 8, 'min_samples_split': 3, 'learning_rate': 0.01}
params_1 = {'n_estimators': 100, 'max_depth': 10, 'min_samples_split': 5, 'class_weight' : {0:1, 1:1000}, 'n_jobs':5}

scaler = StandardScaler()

clfs = [
    LogisticRegression(C=1e5,class_weight= {0:1, 1:2514}, n_jobs=5),

    GaussianNB(),

    ensemble.RandomForestClassifier(**params_1),

    ensemble.ExtraTreesClassifier(**params_1),

    ensemble.GradientBoostingClassifier(**params_0)

    ]
X_train = df_train[numerical_features1].values

y_train = df_train['is_fraud'].values

X_train = scaler.fit_transform(X_train)

X_valid = df_valid[numerical_features1].values
y_valid = df_valid['is_fraud'].values
X_valid_x= scaler.transform(X_valid)
prob_result = []
df_m = []
clfs_fited = []
for clf in clfs:
    clf_name = clf.__class__.__name__
    print(f"{clf_name}:")
    clf.fit(X_train, y_train)
    clfs_fited.append(clf)
    y_pred = clf.predict(X_valid_x)
    prob_pos  = clf.predict_proba(X_valid_x)[:, 1]
    prob_result.append(prob_pos)
    m = confusion_matrix(y_valid, y_pred)
    clf_score = brier_score_loss(y_valid, prob_pos, pos_label=y_valid.max())
    print("\tBrier: %1.5f" % (clf_score))
    print("\tPrecision: %1.5f" % precision_score(y_valid, y_pred))
    print("\tRecall: %1.5f" % recall_score(y_valid, y_pred))
    print("\tF1: %1.5f" % f1_score(y_valid, y_pred))
    print("\tauc: %1.5f" % roc_auc_score(y_valid, prob_pos))
    print("\tAccuracy: %1.5f\n" % accuracy_score(y_valid, y_pred))
    df_m.append(
        pd.DataFrame(m, index=['True Negative', 'True Positive'], columns=['Pred. Negative', 'Pred. Positive'])
        )
joblib.dump(clf, f'{clf_name}.joblib')
joblib.dump(scaler, 'scaler.joblib')

# encoder = LabelEncoder()
# for feature in categorical_features:
#     X[feature] = encoder.fit_transform(X[feature].astype(str))

# Scale numerical features
# scaler = StandardScaler()
# X[numerical_features] = scaler.fit_transform(X[numerical_features])


# rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
# rf_model.fit(X_train, y_train)

# # Train GradientBoostingClassifier
# gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
# gb_model.fit(X_train, y_train)

# # Save models and preprocessing tools
# joblib.dump(rf_model, 'rf_model.joblib')
# joblib.dump(gb_model, 'gb_model.joblib')
# joblib.dump(X_valid_x, 'scaler.joblib')
# # joblib.dump(encoder, 'encoder.joblib')

# # Optionally save the feature dictionary if needed
# feature_dict = {
#     'categorical_features': categorical_features,
#     'numerical_features': numerical_features
# }
# joblib.dump(feature_dict, 'feature_dict.joblib')

# # Evaluate models (optional)
# y_pred_rf = rf_model.predict(X_valid_x)
# y_pred_gb = gb_model.predict(X_valid_x)

# print("Random Forest Classifier")
# print(classification_report(y_valid, y_pred_rf))
# print("Accuracy:", accuracy_score(y_valid, y_pred_rf))

# print("Gradient Boosting Classifier")
# print(classification_report(y_valid, y_pred_gb))
# print("Accuracy:", accuracy_score(y_valid, y_pred_gb))

