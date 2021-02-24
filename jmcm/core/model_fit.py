from math import *
import numpy as np
from scipy.optimize import *

from .base_func import BaseFunc


class ModelFit(BaseFunc):
    """
    Joint mean-covariance models for longitudinal data using MCD method
    """

    def __init__(self, mat_X, mat_Z, mat_W, vec_y, vec_n, optim_meth='default'):
        """
        Basic function for fitting the joint models

        Parameters:
        - mat_X, mat_Z, mat_W, vec_y, vec_n: see notes in BaseFunc
        - optim_meth: The optimisation algorithm used. There are 2 options:
            [1] 'default': use profile likelihood estimation to update
             the 3 types of parameters.
            [2] 'BFGS': use BFGS method to update all 3 parameters together
             If not specified, 'default' would be used
        """
        super().__init__(mat_X, mat_Z, mat_W, vec_y, vec_n)
        self.optim_meth = optim_meth

        # Get the dimension of beta, lambda, gamma
        self._num_bta = self.mat_X.shape[1]
        self._num_lmd = self.mat_Z.shape[1]
        self._num_gma = self.mat_W.shape[1]
        self._num_theta = self._num_bta + self._num_lmd + self._num_gma

        # Predefine a place to store the estimated parameters
        self._theta = np.empty(self._num_theta)
        self._bta = self._theta[:self._num_bta]
        self._lmd = self._theta[self._num_bta:self._num_bta + self._num_lmd]
        self._gma = self._theta[self._num_bta + self._num_lmd:]

        # Fit the model
        self._model_fit()

        self._update_dots()

        # Calculate the likelihood and BIC
        self.max_log_lik, self.bic = self._get_bic(self._theta)

    def _obj_fun_lmd(self, lmd, vec_e):
        """
        minus twice the log likelihood with respect to lambda
        vec_e is a fixed vector when beta and gamma are given
        """
        zlmd = self.mat_Z @ lmd
        return np.sum(zlmd) + np.exp(-zlmd) @ vec_e

    def _grad_lmd(self, lmd, vec_e):
        """
        The first derivative of obj_fun with respect to lambda
        """
        zlmd = self.mat_Z @ lmd
        return self.mat_Z.T @ (1. - np.exp(-zlmd) * vec_e)

    def _obj_fun(self, theta):
        """
        minus twice the log likelihood with respect to theta
        """
        self._xbta = self.mat_X @ theta[0:self._num_bta]
        self._wgma = self.mat_W @ theta[self._num_bta + self._num_lmd:]
        lmd = theta[self._num_bta:self._num_bta + self._num_lmd]
        vec_e = self._get_e()
        return self._obj_fun_lmd(lmd, vec_e)

    def _grad(self, theta):
        """
        The first derivative of obj_fun with respect to theta
        """
        self._xbta = self.mat_X @ theta[0:self._num_bta]
        self._zlmd = self.mat_Z @ theta[self._num_bta:self._num_bta + self._num_lmd]
        self._wgma = self.mat_W @ theta[self._num_bta + self._num_lmd:]

        output = np.zeros(self._num_theta)
        for i in range(self.m):
            inv_Di = self.get_D(i, inverse=True)
            mat_Ti = self.get_T(i)
            inv_Sigma = (mat_Ti.T) @ inv_Di @ mat_Ti
            mat_Xi = self.get_X(i)
            vec_ri = self.get_residual(i)
            output[0:self._num_bta] += mat_Xi.T @ inv_Sigma@vec_ri

            mat_Zi = self.get_Z(i)
            vec_tr = mat_Ti @ vec_ri
            vec_ei = vec_tr**2
            output[self._num_bta:self._num_bta + self._num_lmd] += mat_Zi.T @ (np.diag(inv_Di) * vec_ei - 1.) / 2

            mat_Gi = self._get_G(i)
            output[self._num_bta + self._num_lmd:] += (np.diag(inv_Di) * vec_tr) @ mat_Gi
        return output * (-2)

    def _update_beta(self):
        """
        Calculate beta given lambda and gamma
        """
        # Initialise sums with zero matrix and zero vector
        xsx = np.zeros((self._num_bta, self._num_bta))
        xsy = np.zeros((self._num_bta))
        for i in range(self.m):
            # Select the i-th subject
            yi = self.get_y(i)
            mat_Xi = self.get_X(i)
            # Compute inv_Sigma
            inv_Sigma = self.get_Sigma(i, inverse=True)
            # Compute the sums in the loop
            xs = mat_Xi.T @ inv_Sigma
            xsx += xs @ mat_Xi
            xsy += xs @ yi
        self._theta[:self._num_bta] = np.linalg.inv(xsx) @ xsy

    def _update_lambda(self):
        """
        calculate lambda given beta and gamma
        """
        # give vec e as an argument in the optimisation
        vec_e = self._get_e()
        # use package 'scipy.optimize' and find the minimum
        res = minimize(fun=self._obj_fun_lmd, x0=self._lmd,
                       args=vec_e, method='BFGS', jac=self._grad_lmd)
        self._theta[self._num_bta:self._num_bta + self._num_lmd] = res.x

    def _update_gamma(self):
        """
        Calculate beta given beta and lambda
        """
        # Initialise sums with zero matrix and zero vector
        gdg = np.zeros((self._num_gma, self._num_gma))
        gdr = np.zeros((self._num_gma))
        for i in range(self.m):
            # Compute r, G, and inv_D
            ri = self.get_residual(i)
            mat_Gi = self._get_G(i)
            inv_Di = self.get_D(i, inverse=True)
            # Compute the sums in the loop
            gd = mat_Gi.T @ inv_Di
            gdg += gd @ mat_Gi
            gdr += gd @ ri
        self._theta[self._num_bta + self._num_lmd:] = np.linalg.inv(gdg) @ gdr

    def _update_para(self):
        """
        Update the parameters using profile likelihood estimation
        """
        # update beta
        self._update_beta()
        self._xbta = self.mat_X @ self._bta
        # update lambda
        self._update_lambda()
        self._zlmd = self.mat_Z @ self._lmd
        # update gamma
        self._update_gamma()
        self._wgma = self.mat_W @ self._gma

    def _update_dots(self):
        """
        Update the linear predictors
        """
        self._xbta = self.mat_X @ self._bta
        self._zlmd = self.mat_Z @ self._lmd
        self._wgma = self.mat_W @ self._gma

    def _auto_initial(self):
        """
        Give initial values for beta, lambda, gamma
        by fitting linear regression models
        """
        # initialise beta
        self._theta[:self._num_bta] = self._linear_fit(self.mat_X, self.vec_y)
        # initialise lambda
        fit_r = self.vec_y - self.mat_X @ self._bta
        # use log squared residuals as response
        log_var = np.log(fit_r**2)
        self._theta[self._num_bta:self._num_bta +
                    self._num_lmd] = self._linear_fit(self.mat_Z, log_var)
        # initialise gamma with zeros
        self._theta[self._num_bta + self._num_lmd:] = np.zeros(self._num_gma)
        self._update_dots()

    def _method_1(self, tolerance=1e-4, step_max=100):
        """
        Estimate the parameters using profile estimation
        """
        # Initialise parameters
        self._auto_initial()

        # Do the iterations unless the number of
        # steps reaches step_max OR it has converged
        for step in range(step_max):

            # Store the last values of the parameters
            theta_old = self._theta.copy()
            # Update the parameters
            self._update_para()
            # Get the search direction
            direc = self._theta - theta_old
            # Get the step length
            alpha = self._get_step(self._obj_fun, self._grad, theta_old, direc)
            # Update the parameters again via line search method
            direc *= alpha
            self._theta[:] = theta_old + direc
            self._update_dots()

            # Convergence conditions to stop the iteration
            scale = lambda x: np.maximum(np.absolute(x), 1.)
            # if the direction is small enough
            tol = np.max(np.absolute(direc) / scale(self._theta))
            if tol < tolerance:
                break
            # if the gradiant is close to 0
            f = self._obj_fun(self._theta)
            g = self._grad(self._theta)
            grad_tol = np.max(np.absolute(g) * scale(self._theta) / scale(f))
            if grad_tol < tolerance:
                break

    def _method_2(self):
        """
        Estimate the parameters using BFGS algorithm
        """
        # Initialise parameters
        self._auto_initial()

        res = minimize(fun=self._obj_fun, x0=self._theta,
                       method='BFGS', jac=self._grad)
        self._theta[:] = res.x

    def _model_fit(self):
        """
        Estimate the parameters
        """
        # Choose optimisation method
        if self.optim_meth == 'default':
            self._method_1()
        elif self.optim_meth == 'BFGS':
            self._method_2()
        else:
            raise NameError(
                "The optimal method should be either 'default' or 'BFGS'.")

    def _get_bic(self, theta):
        """
        Calculate the maximum of log likelihood and BIC
        """
        max_log_lik = - self._obj_fun(theta) / 2
        bic = (len(theta) * log(self.m) - 2 * max_log_lik) / self.m
        return max_log_lik, bic
