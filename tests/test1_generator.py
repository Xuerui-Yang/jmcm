import pytest
import pandas as pd

from data_generator import simulate,sim_par

def test():
    m=3
    bta,lmd,gma=sim_par(3,3,3)
    assert len(bta)==3
    df=simulate(bta,lmd,gma,m)
    assert isinstance(df, pd.DataFrame)
    assert df.shape[1]==3
    assert df.iloc[-1]['id']==m-1

if __name__ == '__main__':
    pytest.main(["test1_generator.py"])
