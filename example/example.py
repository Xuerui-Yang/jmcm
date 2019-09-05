import pandas as pd

from jmcm import JointModel

### import the data
diabetes = pd.read_table("data_new.txt")

### fit the model
JM=JointModel(diabetes, 'y|patid|time~1|1', (3,3,3),optim_meth='default')

### summarise the estimation
JM.summary()

#JM.wald_test()

#JM.boot_curve(10)

