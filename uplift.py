import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
# T-learner style
df = pd.read_csv('data/synthetic/aid_retention.csv')
treat = df[df['aid_offer']==1]; control = df[df['aid_offer']==0]
feat = ['gpa','need_index']
X_t, y_t = treat[feat], treat['retained']
X_c, y_c = control[feat], control['retained']
mt = RandomForestClassifier(n_estimators=300, random_state=42).fit(X_t,y_t)
mc = RandomForestClassifier(n_estimators=300, random_state=42).fit(X_c,y_c)
# individual uplift = P(retained|aid=1) - P(retained|aid=0)
df['uplift_est'] = mt.predict_proba(df[feat])[:,1] - mc.predict_proba(df[feat])[:,1]
df[['gpa','need_index','aid_offer','retained','uplift_est']].to_csv('uplift_scored.csv', index=False)
print('Saved uplift_scored.csv (higher = more expected benefit from aid)')