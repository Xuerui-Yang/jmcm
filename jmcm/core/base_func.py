import numpy as np
from numpy.linalg import *
from math import *
from scipy.optimize import *
import warnings



class BaseFunc():
    """
    Include functions to get the sub design matrices for the specific indix
    Include line search method
    Include functions to compute the information matrices
    """

    def __init__(self, mat_X, mat_Z, mat_W, vec_y, vec_n):
        """
        Define variables of use

        Parameters:
        - mat_X: The design matrix for the mean parameter beta
        - mat_Z: The design matrix for the innovation parameter lambda
        - mat_W: The design matrix for the auto regressive parameters gamma
        - vec_y: The response vextor
        - vec_n: The vector of the number of measurements for each subject
        """
        self.mat_X = mat_X
        self.mat_Z = mat_Z
        self.mat_W = mat_W
        self.vec_y = vec_y
        self.vec_n = vec_n

        # The number of subjects
        self.m = len(vec_n)
        # An index list for searching the indices of X and Z
        self._index_list = self._get_index_list(self.vec_n)
        # Calculate the number of rows of W that the i-th subject has
        f = lambda x: x * (x - 1) / 2
        vec_nW = f(self.vec_n)
        # An index list for searching the indices of W
        self._w_index_list = self._get_index_list(vec_nW)

        # Vectors for storing the linear predictors
        self._xbta = None
        self._zlmd = None
        self._wgma = None

    def _get_index_list(self, numbers):
        """
        Given the number of measurements for each subject, return
        a list of indices
        """
        # Cumulative sums
        res = np.cumsum(numbers, dtype='int32')
        # Add 0
        res = np.concatenate(([0], res))
        return res

    def _get_index(self, i):
        """
        For the i-th subject, give its first index and last index
        """
        return self._index_list[i], self._index_list[i + 1]

    def _get_index_W(self, i):
        """
        For the i-th subject, return its irst index (and last index) for
        matrix W
        """
        return self._w_index_list[i], self._w_index_list[i + 1]

    def get_X(self, i):
        # Get matrix X for i-th subject
        first, last = self._get_index(i)
        return self.mat_X[first:last]

    def get_Z(self, i):
        # Get matrix Z for i-th subject
        first, last = self._get_index(i)
        return self.mat_Z[first:last]

    def get_W(self, i):
        # Get matrix W for i-th subject
        first, last = self._get_index_W(i)
        return self.mat_W[first:last]

    def get_y(self, i):
        """
        Get the response for the i-th subject
        """
        first, last = self._get_index(i)
        return self.vec_y[first:last]

    def get_mean(self, i):
        """
        Calculate the mean vector mu_i
        """
        if self._xbta is not None:
            # Get the index of the measurements for i-th subject
            first, last = self._get_index(i)
            return self._xbta[first:last]
        else:
            raise ValueError("Vector bta has NOT been computed.")

    def get_D(self, i, inverse=False):
        """
        Calculate the diagonal matrix D_i contains innovation variances
        When inverse is True, get the inversed matrix D
        """
        if self._zlmd is not None:
            # Get the index of the measurements for i-th subject
            first, last = self._get_index(i)
            # Log diagonal entries are the matrix product of Z and lambda
            if inverse == True:
                log_diag = -self._zlmd[first:last]
            else:
                log_diag = self._zlmd[first:last]
            mat = np.diag(np.exp(log_diag))
            return mat
        else:
            raise ValueError("Vector lambda has NOT been computed.")

    def get_T(self, i, inverse=False):
        """
        Calculate the lower triangular matrix T_i contains generalsed
        autogressive parameters
        """
        if self._wgma is not None:
            ni = self.vec_n[i]
            # Ti=1 for the subjects with only 1 measurement
            if ni == 1:
                return np.array([[1.]])
            else:
                # Get the index of W for i-th subject
                first, last = self._get_index_W(i)
                # Get the generalsed autogressive parameters
                gen_aut_par = self._wgma[first:last]
                # Initialise T with indentity matrix
                mat = np.eye(ni)
                # Add lower triangular entries
                index = 0
                for j in range(1, ni):
                    for k in range(j):
                        mat[j, k] = -gen_aut_par[index]
                        index += 1
                # When inv T_i is needed instead of T_i
                if inverse == True:
                    return np.linalg.inv(mat)
                else:
                    return mat
        else:
            raise ValueError("Vector gamma has NOT been computed.")

    def get_residual(self, i):
        """
        Get the residuals of yi, which is equal to yi-X_i@bta
        """
        yi = self.get_y(i)
        mui = self.get_mean(i)
        return yi - mui

    def get_Sigma(self, i, inverse=False):
        """
        Calculate the covariance matrix Sigma_i
        When inverse is True, get the inversed Sigma
        """
        # Get matrix T and inv matrix D when inverse is True
        # Or inv matrix T and matrix D when inverse is False
        mat_A = self.get_T(i, not(inverse))
        mat_B = self.get_D(i, inverse)
        # Compute Sigma or inversed Sigma
        if inverse:
            return (mat_A.T) @ mat_B @ mat_A
        else:
            return mat_A @ mat_B @ (mat_A.T)

    def _get_G(self, i):
        """
        Calculate matrix G
        """
        # Get the size of G and initialise with zeros
        num_row_G = self.vec_n[i]
        num_col_G = self._num_gma
        mat_G = np.zeros((num_row_G, num_col_G))
        if num_row_G == 1:
            pass
        else:
            # Get the index of the first row of matrix Wi
            index = self._get_index_W(i)[0]
            # Get vector ri
            vec_ri = self.get_residual(i)
            # Update the rows of G
            for j in range(1, num_row_G):
                mat_G[j, :] += vec_ri[0:j] @ self.mat_W[index:index + j, :]
                index += j
        return mat_G

    def _get_e(self):
        """
        Calculate e, which is the squared autoregressive residuals
        """
        # T@r gives the autoregressive residuals
        vec_e = np.empty(0)
        for i in range(self.m):
            mat_T = self.get_T(i)
            vec_r = self.get_residual(i)
            vec_e = np.concatenate((vec_e, (mat_T@vec_r)**2))
        return vec_e

    def _linear_fit(self, x, y):
        """
        Compute the linear regression coefficient
        """
        return np.linalg.inv((x.T @ x)) @ x.T @ y

    def _get_step(self,func, grad, x_old, direc):
        """
        Get the line search length alpha
        """
        # Take the warnings as errors
        warnings.filterwarnings("error")
        try:
            # Do line search using package scipy.optimize
            res = line_search(func, grad, x_old, direc)
        except linesearch.LineSearchWarning:
            # If the iteration Fails, use full step
            alpha = 1.
        else:
            alpha = res[0]
        finally:
            return alpha

    def _info_mat(self):
        """
        Get the information matrices
        """
        # Initialise with zeros
        i_11 = np.zeros((self._num_bta, self._num_bta))
        i_22 = np.zeros((self._num_lmd, self._num_lmd))
        i_33 = np.zeros((self._num_gma, self._num_gma))
        # Update i's in loop
        for i in range(self.m):
            # Compute the matrix for beta
            inv_Di = self.get_D(i, inverse=True)
            mat_Ti = self.get_T(i)
            inv_Sigma = (mat_Ti.T) @ inv_Di @ mat_Ti
            mat_Xi = self.get_X(i)
            i_11 += mat_Xi.T @ inv_Sigma @ mat_Xi

            # Compute the matrix for lambda
            mat_Zi = self.get_Z(i)
            i_22 += mat_Zi.T@mat_Zi / 2 * self.vec_n[i]

            # compute the matrix for gamma
            mat_Wi = self.get_W(i)
            mat_Sigma = np.linalg.inv(inv_Sigma)
            index = 0
            for j in range(1, self.vec_n[i]):
                for k in range(j - 1):
                    wijk = np.expand_dims(mat_Wi[index + k], axis=1)
                    for l in range(j - 1):
                        wijl = np.expand_dims(mat_Wi[index + l], axis=0)
                        i_33 += inv_Di[j, j] * mat_Sigma[k, l] * wijk @ wijl
                index += j
        return i_11, i_22, i_33
