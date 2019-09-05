import pandas as pd
import numpy as np
from jmcm import JointModel
"""
### data preprocessing
def ctr_std(arr):
    sd=np.std(arr)
    mn=np.mean(arr)
    return (arr-mn)/sd

time=ctr_std(df.loc[:,"edate"])
gender=df.loc[:,"gender"]-1
age=df.loc[:,"eventdate"].apply(f)-df.loc[:,"yearofbirth"]
bmi=df.loc[:,"bmi"]
y=1/df.loc[:,"hba1c"]

data=pd.concat([patid,gender,time,age,bmi,y],axis=1)
data.columns=['patid','gender','time','age','bmi','y']

data.to_csv('data_new.txt', header=True, index=False, sep='\t', mode='a')
"""

### import the data
diabetes = pd.read_table("data_new.txt")

### fit the model
JM=JointModel(diabetes, 'y|patid|time~1|1', (3,3,3),optim_meth='default')

### summarise the estimation
JM.summary()

#JM7.wald_test()

#JM.boot_curve(10)

