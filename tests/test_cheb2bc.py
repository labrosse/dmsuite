import numpy as np
import dmsuite as dm

def test_cheb2bc5():
    """ Test of order 5 chebychev diff with robin BC"""
    bc1 =  [[1, 0, 1], [1, 0, 1]]
    bc2 =  [[0, 1, 1], [0, 1, 1]]
    bc3 =  [[1, 0, 1], [0, 1, 1]]
    bc4 =  [[0, 1, 1], [1, 0, 1]]

    computed = dm.cheb2bc(5, bc1)
    for ii in range(5):
        expected = np.load('tests/data/cheb2bc1_5_' + np.str(ii) + '.npy')
        assert np.allclose(computed[ii], expected)

    computed = dm.cheb2bc(5, bc2)
    for ii in range(5):
        expected = np.load('tests/data/cheb2bc2_5_' + np.str(ii) + '.npy')
        assert np.allclose(computed[ii], expected)

    computed = dm.cheb2bc(5, bc3)
    for ii in range(5):
        expected = np.load('tests/data/cheb2bc3_5_' + np.str(ii) + '.npy')
        assert np.allclose(computed[ii], expected)

    computed = dm.cheb2bc(5, bc4)
    for ii in range(5):
        expected = np.load('tests/data/cheb2bc4_5_' + np.str(ii) + '.npy')
        assert np.allclose(computed[ii], expected)

    
    

