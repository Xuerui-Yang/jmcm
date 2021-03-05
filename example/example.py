import pandas as pd

from jmcm import JointModel

### import the data
df=pd.read_csv("aids.csv")
df['cd4s']=np.sqrt(df['cd4'])

### fit the model
JM=JointModel(df, 'cd4s|id|time~1|1',(8,3,3), optim_meth='default')

### summarise the estimation
JM.summary()

#JM.wald_test()

#JM.boot_curve(10)

