import pytest
from experiments.mnistdrawing1_0.environnment import Canvas


def test_calc_dir_length():
    env = Canvas()
    
    assert env._calc_dir_length(1) == (1, 1, True)
    assert env._calc_dir_length(5) == (1, 2, True)
    
    assert env._calc_dir_length(4) == (4, 1, True)
    assert env._calc_dir_length(8) == (4, 2, True)
    
    with pytest.raises(ValueError):
        env._calc_dir_length(0)
        env._calc_dir_length(201)

def test_calc_dir_length_nodraw():
    env = Canvas()
    
    assert env._calc_dir_length(-1) == (1, 1, False)
    assert env._calc_dir_length(-5) == (1, 2, False)
    
    assert env._calc_dir_length(-4) == (4, 1, False)
    assert env._calc_dir_length(-8) == (4, 2, False)
    
    with pytest.raises(ValueError):
        env._calc_dir_length(0)
        env._calc_dir_length(-201)
