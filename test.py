import numpy as np
import matplotlib.pyplot as plt
import ilfft_interp as bf
import numpy.linalg as lin
import numpy.random as rand

def test_transform(lat, f):
    # Test transform invertibility with a specific function.
    # Good to see if the problem is a specific set of coefficients
    lat.eval_func(f)
    A = [L.copy() for L in lat.grids]
    lat.transform()
    lat.inverse_transform()
    for i in range(len(lat.grids)):
        print(lin.norm(A[i] - lat.grids[i]))

def test_invertibility(lat):
    # Test that transform and inverse transform are actually inverse
    g_list = []
    for g in lat.grids:
        g[...] = rand.rand(*g.shape)
        g_list.append(g.copy())
    lat.transform()
    lat.inverse_transform()
    for i in range(len(g_list)):
        # Should print order epsilon reals
        print(lin.norm(g_list[i] - lat.grids[i]))
    return g_list

def test_deriv(lat, ax, f, df):
    lat.eval_func(df)
    g_list = [g.copy() for g in lat.grids]
    
    lat.eval_func(f)
    lat.transform()
    lat.get_deriv(ax)
    lat.inverse_transform()
    for i in range(len(g_list)):
        lat.grids[i] -= g_list[i]
        lat.grids[i][...] = lat.grids[i]**2
    lat.apply_chebyshev_weights()
    return sum([np.sum(L) for L in lat.grids])

def test_integral(lat, f, true_int):
    lat.eval_func(f)
    return np.abs(true_int - lat.get_integral())

def test_int_accuracy(lat):
    for i in range(len(lat.grids)):
        for ind in np.ndindex(lat.grids[i].shape):
            t_d, fac = lat.get_degree(i, ind)
            d = tuple([k for k in t_d[0]])
            t_d = np.array(t_d)
            t_d = (2 - 2*(t_d%2))/(1 - (2*(t_d//2))**2)
            fac = np.array(fac)
            t_d = np.prod(t_d, axis=1)
            I = np.sum(fac*t_d)
            
            def T(*gs):
                val = np.ones(gs[0].shape)
                for j in range(len(d)):
                    val *= np.sign(fac[0])*np.cos(d[j]*np.arccos(gs[j]))
                return val

            lat.eval_func(T)
            print(d, I, lat.get_integral())

