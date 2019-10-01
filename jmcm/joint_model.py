import numpy as np
from tabulate import tabulate
import matplotlib.pyplot as plt
from scipy.stats import chi2
from .core import *

# Set the font size
plt.rcParams.update({'font.size': 14})

class JointModel(ReadData, ModelFit, BaseFunc):

    def __init__(self, df, formula, poly_orders=(), optim_meth='default', model_select=False):
        """
        Basic function for fitting the joint models

        Parameters:
        - df: The dataset of interest. It should be a 'pandas' DataFrame
        - formula: A formula showing the response, subject id, design
             matrices. It is a string following the rules defined in R,
             for example, 'y|id|t~x1+x2|x1+x3'.
            [1] On the left hand side of '~', there are 3 headers of the
             data, partitioned by '|':
             - y: The response vector for all the subjects
             - id: The ID number which identifying different subjects
             - t: The vector of time, which constructs the polynomials
                 for modelling the means, innovation variances, and
                 generalised auto regressive parameters.
            [2] On the right hand side of '~', there are two parts
             partitioned by '|':
             - x1+x2: '+' joints two headers, which are the covariates
                 for the mean. There can be more headers and operators.
                 Package 'patsy' is used to achieve that.
             - x1+x3: Similar to the left part, except that they are
                 for the innovation variances.
        - poly_orders: A tuple of length 3 or length 0. If the length is 3,
             it specifies the polynomial orders of time for the mean, innovation
             variance, and generalised auto regressive parameters. If the length
             is 0, then the model selection procedures might be used.
        - optim_meth: The optimisation algorithm used. There are 2 options:
            [1] 'default': use profile likelihood estimation to update
             the 3 types of parameters.
            [2] 'BFGS': use BFGS method to update all 3 parameters together
             If not specified, 'default' would be used
        - model_select: True or False. To do model selection or not.
        """

        # Read data to get the design matrices, response, and ID numbers
        ReadData.__init__(self, df, formula, poly_orders)
        # Get functions of use
        BaseFunc.__init__(self, self.mat_X, self.mat_Z,
                          self. mat_W, self.vec_y, self.vec_n)

        self.formula = formula
        self.poly_orders = poly_orders
        self.optim_meth = optim_meth

        # Check if model selection precedure is required
        if (isinstance(self.poly_orders, tuple) and
                len(self.poly_orders) == 3 and
                all(isinstance(n, int) for n in self.poly_orders)):
            if model_select == False:
                # just fit the model
                pass
            else:
                # if the request of selection is given and the polynomial
                # orders are given correctly, then traverse under the
                # polynomial orders
                ReadData.__init__(self, df, formula, ())
                self._traverse_models()
                self.mat_X, self.mat_Z, self.mat_W = self._new_design(
                    self.poly_orders)

        else:
            # if the given polynomial orders is not in the correct form
            self.poly_orders = self._model_select()
            self.mat_X, self.mat_Z, self.mat_W = self._new_design(
                self.poly_orders)

        ModelFit.__init__(self, self.mat_X, self.mat_Z,
                          self. mat_W, self.vec_y, self.vec_n, self.optim_meth)

    def summary(self):
        """
        Return the estimated values, maximum log likelihood, and BIC
        """
        print('Model:')
        print(self.formula, self.poly_orders)
        print('Mean Parameters:')
        print(self._bta)
        print('Innovation Variance Parameters:')
        print(self._lmd)
        print('Autoregressive Parameters:')
        print(self._gma)
        print('Maximum of log-likelihood:', self.max_log_lik)
        print('BIC:', self.bic)
        print('')

    def wald_test(self):
        """
        Do the hypothesis test for each parameter
        """
        i_11, i_22, i_33 = self._info_mat()
        inv_11 = np.linalg.inv(i_11)
        inv_22 = np.linalg.inv(i_22)
        inv_33 = np.linalg.inv(i_33)

        print('Wald Test')

        print('Mean Parameters:')
        table = []
        for i in range(self._num_bta):
            est = self._bta[i]
            test = est**2 / inv_11[i, i]
            p_val = chi2.sf(test, 1)
            table.append(['beta' + str(i), est, test, p_val])
        print(tabulate(table, headers=[
              '', 'Estimate', 'chi-square', 'p-value']))
        print('')
        print('Innovation Variance Parameters:')
        table = []
        for i in range(self._num_lmd):
            est = self._lmd[i]
            test = est**2 / inv_22[i, i]
            p_val = chi2.sf(test, 1)
            table.append(['lambda' + str(i), est, test, p_val])
        print(tabulate(table, headers=[
              '', 'Estimate', 'chi-square', 'p-value']))
        print('')
        print('Autoregressive Parameters:')
        table = []
        for i in range(self._num_gma):
            est = self._gma[i]
            test = est**2 / inv_33[i, i]
            p_val = chi2.sf(test, 1)
            table.append(['gamma' + str(i), est, test, p_val])
        print(tabulate(table, headers=[
              '', 'Estimate', 'chi-square', 'p-value']))
        print('')

    def _new_design(self, poly_orders):
        """
        Get the design matrices when doing model selection
        """
        lhs = np.ones((len(self.vec_t), poly_orders[0] + 1))
        for i in range(1, poly_orders[0] + 1):
            lhs[:, i] = np.power(self.vec_t, i)
        mat_X = np.hstack((lhs, self.mat_X))

        lhs = np.ones((len(self.vec_t), poly_orders[1] + 1))
        for i in range(1, poly_orders[1] + 1):
            lhs[:, i] = np.power(self.vec_t, i)
        mat_Z = np.hstack((lhs, self.mat_Z))

        mat_W = self._build_mat_W(poly_orders[2] + 1)

        return mat_X, mat_Z, mat_W

    def _model_select(self):
        """
        Model selection method in finding the best triple of the polynomial orders
        """
        n_max = np.max(self.vec_n)
        mat_X, mat_Z, mat_W = self._new_design((0, n_max - 1, n_max - 1))
        mf = ModelFit(mat_X, mat_Z, mat_W, self.vec_y, self.vec_n)
        temp = mf.bic
        p_star = 0
        for i in range(1, n_max):
            # fit the model with such polynomial orders
            poly_orders = (i, n_max - 1, n_max - 1)
            mat_X, mat_Z, mat_W = self._new_design(poly_orders)
            mf = ModelFit(mat_X, mat_Z, mat_W, self.vec_y, self.vec_n)
            # find the smallest BIC and p*
            if mf.bic < temp:
                temp = mf.bic
                p_star = i

        mat_X, mat_Z, mat_W = self._new_design((n_max - 1, 0, n_max - 1))
        mf = ModelFit(mat_X, mat_Z, mat_W, self.vec_y, self.vec_n)
        temp = mf.bic
        d_star = 0
        for j in range(1, n_max):
            # fit the model with such polynomial orders
            poly_orders = (n_max - 1, j, n_max - 1)
            mat_X, mat_Z, mat_W = self._new_design(poly_orders)
            mf = ModelFit(mat_X, mat_Z, mat_W, self.vec_y, self.vec_n)
            # find the smallest BIC and d*
            if mf.bic < temp:
                temp = mf.bic
                d_star = j

        mat_X, mat_Z, mat_W = self._new_design((n_max - 1, n_max - 1, 0))
        mf = ModelFit(mat_X, mat_Z, mat_W, self.vec_y, self.vec_n)
        temp = mf.bic
        q_star = 0
        for k in range(1, n_max):
            # fit the model with such polynomial orders
            poly_orders = (n_max - 1, n_max - 1, k)
            mat_X, mat_Z, mat_W = self._new_design(poly_orders)
            mf = ModelFit(mat_X, mat_Z, mat_W, self.vec_y, self.vec_n)
            # find the smallest BIC and q*
            if mf.bic < temp:
                temp = mf.bic
                q_star = k

        return (p_star, d_star, q_star)

    def _traverse_models(self):
        p_max, d_max, q_max = self.poly_orders
        mat_X, mat_Z, mat_W = self._new_design((0, 0, 0))
        mf = ModelFit(mat_X, mat_Z, mat_W, self.vec_y, self.vec_n)
        temp = mf.bic
        for i in range(p_max + 1):
            for j in range(d_max + 1):
                for k in range(q_max + 1):
                    print(i, j, k)
                    mat_X, mat_Z, mat_W = self._new_design((i, j, k))
                    mf = ModelFit(mat_X, mat_Z, mat_W, self.vec_y, self.vec_n)
                    if mf.bic < temp:
                        temp = mf.bic
                        self.poly_orders = (i, j, k)

    def _bootstrap_data(self):
        """
        Generate bootstrap sample from data
        """
        index_boot = np.random.choice(self.m, self.m, replace=True)
        y_boot = np.empty(0)
        n_boot = self.vec_n[index_boot]
        x_boot = np.empty((0, self.mat_X.shape[1]))
        z_boot = np.empty((0, self.mat_Z.shape[1]))
        w_boot = np.empty((0, self.mat_W.shape[1]))
        for j in index_boot:
            y_boot = np.concatenate((y_boot, self.get_y(j)))
            x_boot = np.concatenate((x_boot, self.get_X(j)), axis=0)
            z_boot = np.concatenate((z_boot, self.get_Z(j)), axis=0)
            w_boot = np.concatenate((w_boot, self.get_W(j)), axis=0)
        return x_boot, z_boot, w_boot, y_boot, n_boot

    def boot_curve(self, num_boot):
        """
        Plot the curves onto time with bootstrap confidence bands

        Parameters:
        - num_boot: The number of bootstrap samples. (One should try with 
         small numbers at the beginning.)
        """
        # Get the explanatory variables time and time lags
        ts = np.linspace(np.min(self.vec_t), np.max(self.vec_t), 100)
        tslag = np.linspace(0, np.max(self.vec_t) - np.min(self.vec_t), 100)

        # Compute the estimated mean
        mat_X_ts = np.ones((len(ts), self._num_bta))
        for i in range(1, self._num_bta):
            mat_X_ts[:, i] = np.power(ts, i)
        yest = mat_X_ts @self._bta
        
        # Compute the estimated log innovation variance
        mat_Z_ts = np.ones((len(ts), self._num_lmd))
        for i in range(1, self._num_lmd):
            mat_Z_ts[:, i] = np.power(ts, i)
        log_s = mat_Z_ts @ self._lmd
        
        # Compute the estimated generalised auto regressive parameters
        mat_W_tslag = np.ones((len(tslag), self._num_gma))
        for i in range(1, self._num_gma):
            mat_W_tslag[:, i] = np.power(tslag, i)
        garp = mat_W_tslag @ self._gma
        
        # Get the bootstrap estimates
        yest_boot = np.empty((100, num_boot))
        logs_boot = np.empty((100, num_boot))
        garp_boot = np.empty((100, num_boot))
        for i in range(num_boot):
            x_boot, z_boot, w_boot, y_boot, n_boot = self._bootstrap_data()
            mf = ModelFit(x_boot, z_boot, w_boot, y_boot,
                          n_boot, self.optim_meth)
            yest_boot[:, i] = mat_X_ts @ mf._bta
            logs_boot[:, i] = mat_Z_ts @ mf._lmd
            garp_boot[:, i] = mat_W_tslag @ mf._gma

        # Get the lower and upper bands
        yest_u = np.quantile(yest_boot, 0.975, axis=1)
        yest_l = np.quantile(yest_boot, 0.025, axis=1)
        logs_u = np.quantile(logs_boot, 0.975, axis=1)
        logs_l = np.quantile(logs_boot, 0.025, axis=1)
        garp_u = np.quantile(garp_boot, 0.975, axis=1)
        garp_l = np.quantile(garp_boot, 0.025, axis=1)

        # Plot the curves
        fig = plt.figure(figsize=(12, 8))
        ax1 = fig.add_subplot(211)
        ax1.plot(ts, yest_l, color='black', linewidth=1, linestyle='dashed')
        ax1.plot(ts, yest, color='black', linewidth=1)
        ax1.plot(ts, yest_u, color='black', linewidth=1, linestyle='dashed')
        ax1.scatter(self.vec_t, self.vec_y, s=5, color='#607c8e')
        ax1.set(xlabel="Time", ylabel='Response')

        ax2 = fig.add_subplot(223)
        ax2.plot(ts, logs_l, color='black', linewidth=1, linestyle='dashed')
        ax2.plot(ts, log_s, color='black', linewidth=1)
        ax2.plot(ts, logs_u, color='black', linewidth=1, linestyle='dashed')
        ax2.set(xlabel="Time", ylabel='Log-Variance')

        ax3 = fig.add_subplot(224)
        ax3.plot(tslag, garp_l, color='black', linewidth=1, linestyle='dashed')
        ax3.plot(tslag, garp, color='black', linewidth=1)
        ax3.plot(tslag, garp_u, color='black', linewidth=1, linestyle='dashed')
        ax3.set(xlabel="Time Lag", ylabel='Autoregres. Coeffic.')

        plt.show()
