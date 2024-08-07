from weights_system import get_weights
import numpy as np

def test_get_weights_paper_equality():
    #paperweights = [0.0454,0.0476,0.1221]
    expected_weights = [0.0454,0.0477,0.1221]
    u = [np.array([(-1.0,0.0),(1.0,0.0),(0.0,-1.0),(0.0,1.0)\
                  ,(-1.0,1.0),(1.0,1.0),(1.0,-1.0),(-1.0,-1.0)\
                      ,(-2.0,1.0),(-1.0,2.0),(1.0,2.0),(2.0,1.0)\
                          ,(2.0,-1.0),(1.0,-2.0),(-1.0,-2.0),(-2.0,-1.0)])]
    lengths = np.linalg.norm(u,axis=-1)
    weights = (get_weights(u, lengths)[0]).round(decimals=4)
    unique_weights = np.unique(weights)
    assert np.allclose(expected_weights,unique_weights)
    
def test_get_weights_size():
    n,m = 100,24
    rng = np.random.default_rng()
    u = rng.standard_normal((n,m,2))
    lengths = np.linalg.norm(u,axis=-1)
    weights = np.array(get_weights(u,lengths))
    assert weights.shape == (n,m)
    
if __name__ == "__main__":
    test_get_weights_paper_equality()
    test_get_weights_size()