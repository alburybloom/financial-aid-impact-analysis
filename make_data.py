import pandas as pd, numpy as np, os
os.makedirs('data/synthetic', exist_ok=True)
np.random.seed(33)
n=6000
gpa = np.random.uniform(2.0,4.0,size=n)
need_index = np.random.uniform(0,1,size=n)
aid = (np.random.uniform(0,1,size=n) < (0.3 + 0.3*need_index)).astype(int)  # higher need -> more aid
base_logit = -1.2 + 0.8*(gpa-3.0) - 0.5*need_index
uplift = 0.6*aid*(0.5 + 0.5*need_index)  # aid helps more for higher-need
p = 1/(1+np.exp(-(base_logit + uplift)))
retained = (np.random.uniform(0,1,size=n) < p).astype(int)
pd.DataFrame({'gpa':gpa,'need_index':need_index,'aid_offer':aid,'retained':retained}).to_csv('data/synthetic/aid_retention.csv', index=False)
print('Wrote data/synthetic/aid_retention.csv')