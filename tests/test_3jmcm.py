import pytest

from data_generator import generator
from jmcm import JointModel

def test_modelselection():
	df=generator(5)
	JM=JointModel(df, formula='y|id|t~1|1',poly_orders=(2,2,2),model_select=True,optim_meth='default')

def test_modelfit():
	df=generator(5)
	JM0=JointModel(df, formula='y|id|t~1|1',poly_orders=(0,0,0),optim_meth='default')
	JM1=JointModel(df, formula='y|id|t~1|1',poly_orders=(0,0,0),optim_meth='BFGS')

def test_functions():
	df=generator(5)
	JM=JointModel(df, formula='y|id|t~1|1',poly_orders=(2,2,2),optim_meth='default')
	JM.summary()
	JM.wald_test()
