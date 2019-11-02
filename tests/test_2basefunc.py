import pytest

from data_generator import generator
from jmcm.core.base_func import BaseFunc
from jmcm.core.read_data import ReadData

def test_class():
	df=generator(10)
	rd=ReadData(df, "y|id|t~1|1", (3,3,3))
	assert rd.mat_X.shape[1]==4
	assert rd.mat_Z.shape[1]==4
	assert rd.mat_W.shape[1]==4
	bf=BaseFunc(rd.mat_X, rd.mat_Z, rd.mat_W, rd.vec_y, rd.vec_n)
	assert bf.get_y(0) is not None
	assert bf.get_X(0) is not None
	assert bf.get_Z(0) is not None
	assert bf.get_W(0) is not None

