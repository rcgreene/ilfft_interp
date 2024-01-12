import numpy as np
import numpy.linalg as lin
import scipy.fft as fft
import matplotlib.pyplot as plt

def cheby_points(n):
    return -np.sin(np.linspace(-np.pi/2, np.pi/2, n))

def edge_slice(ax, dim, index):
    return tuple([slice(None) if i != ax else index for i in range(dim)])

class Lattice:
    """Abstract class to standardize some basic lattice methods."""
    def __init__(self, n):
        self.n = n
        self.init_coords()
        self.init_grids()
        self.init_ordering()

    def init_grids(self):
        """ Produces a set of grids representing the collocation points.
         Each grid is Cartesian but a lattice can be made of multiple grids."""
        self.coords = [np.meshgrid(*x, indexing='ij') for x in self.x]
        self.grids = [np.zeros(L[0].shape) for L in self.coords]
        self.coefs = [np.zeros(L.shape) for L in self.grids]

    def get_point_count(self):
        """Returns the number of distinct lattice points in real space."""
        return sum([L.size for L in self.grids])

    def get_points(self):
        """Produces a list of every collocation point of shape (p_num, dim)"""
        p_list = []
        for x_grid in self.coords:
            p_list.extend([[x_grid[i][ind] for i in range(self.dim)] 
                for ind in np.ndindex(x_grid[0].shape)])
        return np.array(p_list)

    def eval_func(self, f):
        """Evaluate a vectorized function f on the coordinate meshgrids."""
        for i, L in enumerate(self.coords):
            self.grids[i][...] = f(*L)

    def init_ordering(self):
        """Produces a lexical ordering. Necessary for differentiation and
          re-ordering for different lattices."""
        lexical_info = []

        for i, L in enumerate(self.grids):
            for ind in np.ndindex(L.shape):
                d_tup, fac = self.get_degree(i, ind)
                n = len(d_tup)
                for j in range(n):
                # Lexical info tuple is in the following format:
                # 0:dim  : Degree of a Chebychev coefficient
                # dim + 1: Factor that the address in coef should be multiplied by
                # dim + 2: Subarray identifier
                # -dim:  : Index into the coef subarray
                    lexical_info.append(tuple([k for k in d_tup[j]] + [fac[j]] +
                        [i] + [k for k in ind]))
        self.axis_sort = []
        for j in range(self.dim):
            lexical_info.sort(key= lambda u: u[j])
        self.axis_sort.append(lexical_info.copy())
        for j in range(self.dim - 1):
            lexical_info.sort(key = lambda u: u[j])
            self.axis_sort.append(lexical_info.copy())
        
        # Pre-planning for derivative computation and mapping to a full
        # Cartesian grid
        self.d_in = np.zeros(len(lexical_info))
        self.d_out = self.d_in.copy()
        self.max_degree = np.zeros(self.dim, dtype=int)
        self.d_blocks = [[] for i in range(self.dim)]
        for i in range(self.dim):
            k = 0
            self.d_blocks[i].append(k)
            ind = np.zeros(self.dim)
            axes = np.array([j for j in range(self.dim) if j != i])
            for x in self.axis_sort[i]:
                x_a = np.array(x)
                if np.any(ind[axes] != x_a[axes]):
                    ind[:] = x_a[:self.dim]
                    self.d_blocks[i].append(k)
                    self.max_degree[i] = max(self.d_blocks[i][-1] - self.d_blocks[i][-2], self.max_degree[i])
                k += 1
            self.d_blocks[i].append(self.d_in.shape[0])

    def normalize_coefficients(self, factor=.5):
        """This implements a non-standard normalization for DCT output to get Chebyshev coefficients."""
        for i, C in enumerate(self.coefs):
            for j in range(self.dim):
                ind = edge_slice(j, self.dim, 0)
                C[ind] *= factor
                if (np.abs(self.x[i][j][-1]) + np.abs(self.x[i][j][0])) == 2:
                    ind = edge_slice(j, self.dim, -1)
                    C[ind] *= factor

    def apply_chebyshev_weights(self):
        """Normalize the lattice by applying chebyshev weights at each grid point."""
        for i, L in enumerate(self.grids):
            L[...] /= self.lat_size*len(self.grids)/(2**self.dim)
            for j in range(self.dim):
                if (np.abs(self.x[i][j][0]) == 1):
                    ind = edge_slice(j, self.dim, 0)
                    L[ind] *= .5
                if (np.abs(self.x[i][j][-1]) == 1):
                    ind = edge_slice(j, self.dim, -1)
                    L[ind] *= .5

    def plot_basis(self):
        """Produces a scatter plot representing the basis."""
        if (self.dim != 2) and (self.dim != 3): 
            print("This is only supported for 2 and 3 dimensional lattices.")

        p_lists = []
        for i, L in enumerate(self.grids):
            p_lists.append([])
            for ind in np.ndindex(L.shape):
                deg_tup = self.get_degree(i, ind)[0]
                for x in deg_tup:
                    p_lists[-1].append(x[:self.dim])
            p_lists[-1] = np.array(p_lists[-1]).T

        fig = plt.figure()
        pv  = "3d" if self.dim == 3 else None
        ax  = fig.add_subplot(projection=pv)
        for i, ps in enumerate(p_lists):
            ax.scatter(*(x for x in ps), s=20)
        plt.show()


    def to_cartesian(self, fill_val = 0):
        """Maps coefficients onto the smallest Cartesian Grid for which the full basis is supported."""
        cart_grid = np.full(self.max_degree, float("NaN"))
        for x in self.axis_sort[0]:
            if np.isnan(cart_grid[x[:self.dim]]):
                cart_grid[x[:self.dim]] = 0
            cart_grid[x[:self.dim]] += self.coefs[x[self.dim + 1]][x[self.dim + 2:]]*x[self.dim]

        if not np.isnan(fill_val):
            for ind in np.ndindex(cart_grid.shape):
                if np.isnan(cart_grid[ind]):
                    cart_grid[ind] = fill_val

        return cart_grid

    def load_d_in(self, ax, c_in=None):
        """Load data from c_in into a vector for differentiation along axis 'ax'
        c_in default: self.coefs"""
        if c_in is None:
            c_in = self.coefs

        for i, x in enumerate(self.axis_sort[ax]):
            self.d_in[i] += c_in[x[self.dim + 1]][x[self.dim + 2:]]*x[self.dim]
        
    def save_d_out(self, ax, c_out=None):
        """Load data from completed derivative calculation to c_out
        c_out default: self.coefs"""
        if c_out is None:
            c_out = self.coefs

        for L in c_out:
            L[...] = 0
        for i, x in enumerate(self.axis_sort[ax]):
            c_out[x[self.dim + 1]][x[self.dim + 2:]] += self.d_out[i]

    def get_deriv(self, ax, c_in=None, c_out=None):
        """Differentiate along axis ax. By default, this uses replaces the function in
        the coefficient array and replaces it with the derivative. 
        This can be avoided by specifying c_in to change the source or c_out to change
        the destination.
        ax, int: axis along which to differentiate.
        c_in, list of arrays of the same shape as self.coefs: Used to supply input coefficients
        c_out, list of arrays of the same shape as self.coefs: Used to store output coefficients"""
        self.d_out[...] = 0
        self.d_in[...] = 0
        self.load_d_in(ax, c_in=c_in)
        
        blocks = self.d_blocks[ax]
        for i in range(len(blocks) - 1):
            if blocks[i + 1] <= blocks[i] + 1:
                continue
            if blocks[i + 1] == blocks[i] + 2:
                self.d_out[blocks[i]] = self.d_in[blocks[i] + 1]
                continue
            self.d_in[blocks[i]:blocks[i + 1]] *= -np.arange(blocks[i + 1] - blocks[i])
            self.d_out[blocks[i + 1] - 2] = 2*self.d_in[blocks[i + 1] - 1]
            for j in range(blocks[i + 1] - 3, blocks[i] - 1, -1):
                self.d_out[j] = 2*self.d_in[j + 1] + self.d_out[j + 2]
            self.d_out[blocks[i]] /= 2

#        for i in range(len(self.d_in)):
#            print(self.d_in[i], self.d_out[i], self.axis_sort[ax][i])
        self.save_d_out(ax, c_out=c_out)

    def make_integral_stencil(self):
        """ Makes an integral stencil to compute Clenshaw-Curtis quadrature without
         transforming the function. (Making the stencil takes O(N log N) time though.)
         This overwrites the coefficient information. This is automatically called if
         get_integral is called before a stencil has been determined, so it is not necessary
         to call this manually."""
        self.d_out[...] = 0
        for i, x in enumerate(self.axis_sort[0]):
            deg = np.array(x[:self.dim])
            if np.all(deg % 2 == 0):
                v = np.prod(2/(1 - deg**2))
                self.d_out[i] = v*x[self.dim]

        self.save_d_out(0)

        if hasattr(self, "adjoint_transform"):
            self.adjoint_transform()         # For Bravais lattices, this aliases the inverse transform
                                             # The actual operator here is not really the adjoint
                                             # but the difference is only scaling, which is handled
                                             # through the choice of coefficients.
        else:
            # For Bravais lattices, the inverse is the adjoint up to some scaling.
            self.normalize_coefficients(factor=.5)
            self.inverse_transform()
            self.apply_chebyshev_weights()

        self.stencil = [L.copy() for L in self.grids]

    def get_integral(self):
        """Numerically integrate the function stored in self.grids. The first call may be slower than
        subsequent calls if the stencil has not been predetermined."""
        if not hasattr(self, "stencil"):
            g_list = [L.copy() for L in self.grids]
            self.make_integral_stencil()
            for i in range(len(self.grids)):
                self.grids[i][...] = g_list[i]
        return sum([np.sum(self.stencil[i]*self.grids[i]) for i in range(len(self.grids))])

class HexLattice(Lattice):
    """ A 4 by 7 Padua-type 2 dimensional lattice. This is extremely
     close to the optimal Euclidean degree efficiency for a 2d lattice
     without sacrificing easy scalability. 
     n, int: resolution parameter. The lattice has Euclidean degree 4n"""
    def __init__(self, n):
        self.dim = 2
        Lattice.__init__(self, n)
        self.lat_size = self.grids[1].size

    def init_coords(self):
        self.res = [4*self.n + 1, 7*self.n + 1 - self.n%2]
        x = cheby_points(self.res[0])
        y = cheby_points(self.res[1])
        self.x = [(x[::2], y[::2]), (x[1::2], y[1::2])]

    def get_degree(self, i, ind):
        if i == 0:
            return [ind], [1]
        ind_list = np.array([[self.res[0] - 1 - ind[0], ind[1]], [ind[0], self.res[1] - 1 - ind[1]]])
        i_n = lin.norm(ind_list, axis=1)
        if i_n[0] < i_n[1]:
            return [ind_list[0]], [1]
        elif i_n[1] < i_n[0]:
            return [ind_list[1]], [1]
        else:
            return ind_list, [.5, .5]

    def transform(self):
        """ Transform sampled function to Chebyshev coefficients."""
        #There are simpler ways to scale the basis consistently
        #but then the 'coefs' data wouldn't be exactly Chebyshev coefficients.
        self.coefs[0][...] = fft.dctn(self.grids[0], type=1, norm="forward")*4
        self.coefs[1][...] = fft.dctn(self.grids[1], type=2, norm="forward")*4
        self.coefs[0][0,:] /= 2
        self.coefs[0][:,0] /= 2
        self.coefs[0][-1,:] /= 2
        self.coefs[0][:,-1] /= 2
        self.coefs[1][0,:] /= 2
        self.coefs[1][:,0] /= 2
        for ind in np.ndindex(self.coefs[1].shape):
            a = (self.coefs[0][ind] + self.coefs[1][ind])*.5
            self.coefs[1][ind] = (self.coefs[0][ind] - self.coefs[1][ind])*.5
            self.coefs[0][ind] = a

    def inverse_transform(self):
        """Transform coefficients to grid values."""
        for ind in np.ndindex(self.coefs[1].shape):
            a = (self.coefs[0][ind] + self.coefs[1][ind])
            self.coefs[1][ind] = (self.coefs[0][ind] - self.coefs[1][ind])
            self.coefs[0][ind] = a
        self.coefs[0][0,:] *= 2
        self.coefs[0][:,0] *= 2
        self.coefs[0][-1,:] *= 2
        self.coefs[0][:,-1] *= 2
        self.coefs[1][0,:] *= 2
        self.coefs[1][:,0] *= 2
        self.grids[0] = fft.dctn(self.coefs[0], type=1)/4
        self.grids[1] = fft.dctn(self.coefs[1], type=3)/4

class PadLattice(Lattice):
    """An n by n+1 Checkerboard 2 dimensional lattice. Optimal in total degree.
    n, int: Resolution parameter. The total degree of the basis is n - 1"""
    def __init__(self, n):
        self.dim = 2
        Lattice.__init__(self, n)
        self.storage = np.zeros((self.res[0], (self.res[1] + 1)//2))
        self.lat_size = ((self.res[0] - 1)*(self.res[1] - 1))/4

    def init_coords(self):
        self.res = [self.n + self.n%2, self.n + 1 - self.n%2] #res is n x n+1 but first coord is always even
        x = cheby_points(self.res[0])
        y = cheby_points(self.res[1])
        self.x = [(x[::2], y[::2]), (x[1::2], y[1::2])]

    def get_degree(self, i, ind):
        if i == 0:
            return [ind], [1]
        if i == 1:
            if (self.res[0] - ind[0] + ind[1]) > (self.res[1] - ind[1] + ind[0]):
                return [(ind[0], self.res[1] - 1 - ind[1])], [1]
            return [(self.res[0] - 1 - ind[0], ind[1])], [1]

    def transform(self):
        self.storage[1::2, -1] = 0
        self.storage[::2] = fft.dct(self.grids[0], axis=1, type=1)
        self.storage[1::2, :-1] = fft.dct(self.grids[1], axis=1, type=2)
        self.storage[:, 0] /= 2
        self.storage[...] = fft.dct(self.storage, axis=0, type=1)/(2*self.lat_size)
        self.storage[0, :] /= 2
        self.storage[-1, :] /= 2
        self.coefs[0][...] = self.storage[:self.res[0]//2]
        self.coefs[1][...] = self.storage[self.res[0]//2:, :-1]

    def inverse_transform(self):
        self.storage[self.res[0]//2:, -1] = 0
        self.storage[self.res[0]//2:, :-1] = self.coefs[1]
        self.storage[:self.res[0]//2] = self.coefs[0]
        self.storage[0, :] *= 2
        self.storage[-1, :] *= 2
        self.storage[...] = fft.dct(self.storage, axis=0, type=1)/4
        self.storage[:, 0] *= 2
        self.storage[:,-1] *= 2
        self.grids[0][...] = fft.dct(self.storage[::2], axis=1, type=1)
        self.grids[1][...] = fft.dct(self.storage[1::2, :-1], axis=1, type=3)

class BCCLattice(Lattice):
    """A Body-Centered Cubic lattice. The optimal 3d lattice for Euclidean and Total degree
    efficiency.
    n, int: Resolution parameter. The Eucldean degree is sqrt(2)*n"""
    def __init__(self, n):
        self.dim = 3
        Lattice.__init__(self, n)
        self.lat_size = self.grids[1].size

    def init_coords(self):
        self.res = [2*self.n + 1]*3
        x = cheby_points(self.res[0])
        self.x = [(x[::2], x[::2], x[::2]), (x[1::2], x[1::2], x[1::2])]

    def get_degree(self, i, ind):
        if i == 0:
            return [ind], [1]
        if i == 1:
            max_i = np.amax(ind)
            out_list = [tuple([ind[k] if k != j else self.res[0] - 1 - ind[k] for k in range(3)]) 
                        for j in range(3) if ind[j] == max_i]
            return out_list, [1/len(out_list)]*len(out_list)

    def transform(self):
        self.coefs[0] = fft.dctn(self.grids[0], type=1)/self.grids[1].size
        self.coefs[1] = fft.dctn(self.grids[1], type=2)/self.grids[1].size
        self.coefs[0][0,...] /= 2
        self.coefs[0][:,0,:] /= 2
        self.coefs[0][...,0] /= 2
        self.coefs[0][-1,...] /= 2
        self.coefs[0][:,-1,:] /= 2
        self.coefs[0][...,-1] /= 2
        self.coefs[1][0,...] /= 2
        self.coefs[1][:,0,:] /= 2
        self.coefs[1][...,0] /= 2
        for ind in np.ndindex(self.coefs[1].shape):
            a = .5*(self.coefs[0][ind] + self.coefs[1][ind])
            self.coefs[1][ind] = .5*(self.coefs[0][ind] - self.coefs[1][ind])
            self.coefs[0][ind] = a
    
    def inverse_transform(self):
        for ind in np.ndindex(self.coefs[1].shape):
            a = self.coefs[0][ind] + self.coefs[1][ind]
            self.coefs[1][ind] = self.coefs[0][ind] - self.coefs[1][ind]
            self.coefs[0][ind] = a
        self.coefs[0][0,...] *= 2
        self.coefs[0][:,0,:] *= 2
        self.coefs[0][...,0] *= 2
        self.coefs[0][-1,...] *= 2
        self.coefs[0][:,-1,:] *= 2
        self.coefs[0][...,-1] *= 2
        self.coefs[1][0,...] *= 2
        self.coefs[1][:,0,:] *= 2
        self.coefs[1][...,0] *= 2
        self.grids[0] = fft.dctn(self.coefs[0], type=1)/8
        self.grids[1] = fft.dctn(self.coefs[1], type=3)/8

class OctLattice(Lattice):
    # The simplest composite lattice worth using. A seven-point composite stencil with
    # cartesian symmetry
    # WIP
    def __init__(self, n):
        self.dim = 2
        self.alias = n*2 - 2
        self.offset = .587 # Need to check on this again
        Lattice.__init__(self, n)
        self.storage = np.zeros(self.grids[3].shape, dtype=complex)

    def init_coords(self):
        self.res = [self.n]*2
        x_1 = np.linspace(-np.pi/2, 3*np.pi/2, 2*self.res[0] - 1)[:-1]
        h = (x_1[1] - x_1[0])*.5
        x_2 = np.linspace(-np.pi/2, np.pi/2, 2*self.res[0] - 1)[1::2]
        x_3 = x_1 + h*self.offset
        self.x = [(np.sin(x_1[:self.res[0]]), np.sin(x_1[:self.res[0]])),
                (np.sin(x_1[:self.res[0]]), np.sin(x_2)), (np.sin(x_2), np.sin(x_1[:self.res[0]])),
                (np.sin(x_3), np.sin(x_3))]
        

    def get_degree(self, i, ind):
        match i:
            case 3:
                return [ind], [1]
            case 2:
                return [(ind[0], self.alias + ind[1])], [1]
            case 1:
                return [(self.alias + ind[0], ind[1])], [1]
            case 0:
                if ind[0] < ind[1]:
                    return [(self.alias + ind[0], self.alias - ind[1])], [1]
                if ind[1] < ind[0]:
                    return [(self.alias - ind[0], self.alias + ind[1])], [1]
                return [(self.alias + ind[0], self.alias - ind[1]),
                        (self.alias - ind[0], self.alias + ind[1])], [.5, .5]

    def get_vector(self, ind):
        out_list = [(3, ind), (0, ind)]
        if (ind[0] != n - 1):
            out_list.extend([(3, (self.alias - ind[0], ind[1])), (2, ind)])
        if (ind[1] != n - 1):
            out_list.extend([(3, (ind[0], self.alias - ind[1])), (1, ind)])
        if (ind[0] != n - 1) and (ind[1] != n - 1):
            out_list.append((3, (self.alias - ind[0], self.alias - ind[1])))

    def transform(self):
        self.storage[...] = fft.fft3(self.grids[3])/self.grids[3].size
        self.coefs[0] = fft.dctn(self.grids[0], type=1)/(self.n - 1)**2
        self.coefs[1] = fft.dct(self.grids[1], ax=0, type=1)/(self.n - 1)**2
        self.coefs[1] = fft.dct(self.coefs[1], ax=1, type=2)
        self.coefs[2] = fft.dct(self.grids[2], ax=0, type=2)/(self.n - 1)**2
        self.coefs[2] = fft.dct(self.coefs[2], ax=1, type=1)
                


class FCCLattice(Lattice):
    """ A Face-Centered Cubic Lattice. Not optimal in Euclidean or total degree but better than
     a cartesian lattice, has strictly positive integration weights and better efficiency in
     its representation of boundary conditions.
     n, int: Resolution parameter. The Euclidean degree is sqrt(3)*n"""
    def __init__(self, n):
        self.dim = 3
        Lattice.__init__(self, n)
        self.lat_size = self.n**3

    def init_coords(self):
        self.res = [2*self.n + 1]*3
        x = cheby_points(self.res[0])
        x_1 = x[::2]
        x_2 = x[1::2]
        self.x = [(x_1, x_1, x_1), (x_2, x_2, x_1), (x_2, x_1, x_2), (x_1, x_2, x_2)]

    def get_degree(self, i, ind):
        ind_v = [np.array(ind)]
        fac = [1]
        a = i % 2
        b = i //2

        if a == 1:
            ind_v = []
            if ind[0] > ind[1]:
                ind_v = [(2*self.n - ind[0], ind[1], ind[2])]
                fac = [1]
            else:
                ind_v.append((ind[0], 2*self.n - ind[1], ind[2]))
                fac = [1]

                if (ind[2] == self.n) and (ind[0] == ind[1]):
                    ind_v.append((2*self.n - ind[0], ind[1], ind[2]))
                    fac = [.5, .5]
                    return ind_v, fac
        if b == 1:
            tmp = []
            tmp_fac = []
            ind_1 = np.abs(np.array(ind_v[0]) - np.array([2*self.n, 2*self.n, 0]))
            ind_2 = np.array((ind_v[0][0], ind_v[0][1], 2*self.n - ind_v[0][2]))
            if lin.norm(ind_1) < lin.norm(ind_2):
                tmp.append(tuple(ind_1))
                tmp_fac.append(1)
            elif lin.norm(ind_2) < lin.norm(ind_1):
                tmp.append(tuple(ind_2))
                tmp_fac.append(1)
            else:
                tmp.extend([tuple(ind_1), tuple(ind_2)])
                m = sum([j != 0 for j in ind_1])
                n = sum([j != 0 for j in ind_2])
                z = 2**m + 2**n
                tmp_fac.extend([(2**m)/z, (2**n)/z])
            ind_v = tmp
            fac = tmp_fac

        return ind_v, fac

    def transform(self):
        self.coefs[0][...] = fft.dctn(self.grids[0], type=1)/(self.n)**3
        self.coefs[1][...] = fft.dct(self.grids[1], type=1, axis=2)/(self.n)**3
        self.coefs[1][...] = fft.dctn(self.coefs[1], type=2, axes=(0,1))
        self.coefs[2][...] = fft.dct(self.grids[2], type=1, axis=1)/(self.n)**3
        self.coefs[2][...] = fft.dctn(self.coefs[2], type=2, axes=(0,2))
        self.coefs[3][...] = fft.dct(self.grids[3], type=1, axis=0)/(self.n)**3
        self.coefs[3][...] = fft.dctn(self.coefs[3], type=2, axes=(1,2))
        for i in range(4):
            self.coefs[i][0,:,:] /= 2
            self.coefs[i][:,0,:] /= 2
            self.coefs[i][:,:,0] /= 2
            if (i == 0) or (i == 1):
                self.coefs[i][:,:,-1] /= 2
            if (i == 0) or (i == 2):
                self.coefs[i][:,-1,:] /= 2
            if (i == 0) or (i == 3):
                self.coefs[i][-1,:,:] /= 2

        i_vec = [0, 2, 0, 1]
        j_vec = [1, 3, 2, 3]
        dim_vec = [self.coefs[1].shape, (self.n, self.n, self.n), 
                self.coefs[2].shape, (self.n,self.n,self.n)]
        for i in range(4):
            m = i_vec[i]
            n = j_vec[i]
            for ind in np.ndindex(dim_vec[i]):
                a = (self.coefs[m][ind] + self.coefs[n][ind])*.5
                self.coefs[n][ind] = (self.coefs[m][ind] - self.coefs[n][ind])*.5
                self.coefs[m][ind] = a
                if (i == 1) and (ind[1] <= ind[0]):
                    self.coefs[3][ind] *= -1
        
        for ind_2 in np.ndindex((self.n, self.n)):
            ind = (self.n, ind_2[0], ind_2[1])
            a = (self.coefs[0][ind] + self.coefs[3][ind])*.5
            self.coefs[3][ind] = (self.coefs[0][ind] - self.coefs[3][ind])*.5
            self.coefs[0][ind] = a

    def inverse_transform(self):
        
        i_vec = [1, 0, 2, 0]
        j_vec = [3, 2, 3, 1]
        dim_vec = [(self.n, self.n, self.n), 
                self.coefs[2].shape, (self.n,self.n,self.n), self.coefs[1].shape]
        for i in range(4):
            m = i_vec[i]
            n = j_vec[i]
            for ind in np.ndindex(dim_vec[i]):
                if (i == 2) and (ind[1] <= ind[0]):
                    self.coefs[3][ind] *= -1
                a = (self.coefs[m][ind] + self.coefs[n][ind])
                self.coefs[n][ind] = (self.coefs[m][ind] - self.coefs[n][ind])
                self.coefs[m][ind] = a
        
        for ind_2 in np.ndindex((self.n, self.n)):
            ind = (self.n, ind_2[0], ind_2[1])
            a = self.coefs[0][ind] + self.coefs[3][ind]
            self.coefs[3][ind] = self.coefs[0][ind] - self.coefs[3][ind]
            self.coefs[0][ind] = a

        for i in range(4):
            self.coefs[i][0,:,:] *= 2
            self.coefs[i][:,0,:] *= 2
            self.coefs[i][:,:,0] *= 2
            if (i == 0) or (i == 1):
                self.coefs[i][:,:,-1] *= 2
            if (i == 0) or (i == 2):
                self.coefs[i][:,-1,:] *= 2
            if (i == 0) or (i == 3):
                self.coefs[i][-1,:,:] *= 2

        self.grids[0][...] = fft.dctn(self.coefs[0], type=1)/8
        self.coefs[1][...] = fft.dct(self.coefs[1], type=1, axis=2)/8
        self.grids[1][...] = fft.dctn(self.coefs[1], type=3, axes=(0,1))
        self.coefs[2][...] = fft.dct(self.coefs[2], type=1, axis=1)/8
        self.grids[2][...] = fft.dctn(self.coefs[2], type=3, axes=(0,2))
        self.coefs[3][...] = fft.dct(self.coefs[3], type=1, axis=0)/8
        self.grids[3][...] = fft.dctn(self.coefs[3], type=3, axes=(1,2))
