import pytest
import pandas as pd

from data_generator import generator

def test():
    m=3
    df=generator(m)
    assert isinstance(df, pd.DataFrame)
    assert df.shape[1]==3
    assert df.iloc[-1]['id']==m-1

