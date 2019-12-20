import numpy as np
import scipy
import matplotlib.pyplot as plt
import warnings
from scipy.stats import gaussian_kde


class PdfApprox:
    # Function: takes numpy array of values and a name for it. Tries fitting different distributions
    # to data, chooses best one according to Kullback-Leibler divergence metric.
    # Possibility to generate values according to chosen distribution.

    def __init__(self, dist_names=['gamma', 'beta', 'norm', 'pareto', 'maxwell'], points=1000):
        """
        self.model_distr: distribution names to try to fit to data
        self.distr_klval: Kullback-Leibler divergences for each distribution for the last processed column
        self.column_distr: dictionary of format <'columnname': ('distr name', distr_params)
        self.points_num: number of points in the grid to evaluate epdf
        """
        self.model_distr = dist_names
        self.distr_klval = {}
        self.column_distr = {}
        self.points_num = points

    def kl_divergence(self, p, q):
        """
        calculate the Kullback-Leibler divergence between two distributions p and q.
        To avoid log(0), we return 0 if p == 0
        """
        return np.sum(np.where(p != 0, p * np.log(p / q), 0))

    def fit(self, values, name, show_plots=False):
        """
        fit the model_distr distributions to an input array. Calculate KL divergence for each one of them
        Creates entry in self.column_distr
        :param values: values of the column to emulate
        :param name: name of the column
        :param show_plots: whether to show comparative plots
        :returns: tuple (best KL divergence, best approx. distribution name, best parameters). Updates self.column_distr
        """
        xmax = max(values)
        xmin = min(values)
        x = np.linspace(xmin, xmax, self.points_num)

        kernel = gaussian_kde(values)
        estimated_pdf = kernel(x)
        distr_models = {}
        warnings.filterwarnings("ignore")
        for dist_name in self.model_distr:
            dist = getattr(scipy.stats, dist_name)
            ###################################
            param = dist.fit(values)
            # Returns:
            # alpha, loc, beta for gamma, where loc is lower limit; mean = alpha*beta
            # alpha, beta, loc, scale for beta, where loc is lower limit,
            #             scale is higher limit; mean = alpha/(alpha+beta)
            # mu, std for norm
            ###################################
            distr_models[dist_name] = param
            pdf_fitted = dist.pdf(x, *param[:-2], loc=param[-2], scale=param[-1])
            KLd = self.kl_divergence(estimated_pdf, pdf_fitted)
            self.distr_klval[dist_name] = KLd
            if show_plots:
                plt.clf()
                plt.figure(figsize=(6, 6))
                plt.plot(x, pdf_fitted, 'r')
                plt.plot(x, estimated_pdf, 'b')
                plt.hist(values, density=True, bins=30)
                plt.title(dist_name)
                plt.show()
        warnings.filterwarnings("default")

        best_distr_name = min(self.distr_klval, key=lambda k: self.distr_klval[k])
        self.column_distr[name] = (best_distr_name, distr_models[best_distr_name])
        return self.distr_klval, best_distr_name, distr_models[best_distr_name]

    def generate_data(self, colname, numpoints):
        """
        generate numpoints data points for column colname. colname must be a key from self.column_distr
        :param colname: column name for which to generate data
        :param numpoints: number of points to generate
        :return: np.array of length numpoints containing data points generated according to distribution of colname
        """
        if colname not in self.column_distr.keys():
            print('pdfApprox::generate_data(): column named ' + str(colname) + 'is unknown. Aborting.')
            return
        dist = getattr(scipy.stats, self.column_distr[colname][0])
        param = self.column_distr[colname][1]
        result = np.array(dist.rvs(*param[:-2], loc=param[-2], scale=param[-1], size=numpoints))
        return result
