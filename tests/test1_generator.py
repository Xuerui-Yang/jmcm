import pytest
import pandas as pd

from data_generator import generator

def test():
    df=generator(m=3)
    assert isinstance(df, pd.DataFrame)
    assert df.shape[1]==3
    assert df.iloc[-1]['id']==m-1

if __name__ == '__main__':
    pytest.main(["test1_generator.py"])
