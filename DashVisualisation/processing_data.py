
import numpy as np

def sample_parameters(var_1_ind, var_2_ind, means, sds, cor, N):
    # Sample parameters for the each variable then create a list of betters

    cov_ = cor * np.sqrt(sds[var_1_ind]**2 * sds[var_2_ind]**2) # covariance between the two variables with the given correlation

    # set up the covariance matrix for sampling values
    cov_matrix = np.diag(sds)**2
    cov_matrix[var_1_ind, var_2_ind] = cov_
    cov_matrix[var_2_ind, var_1_ind] = cov_


    # sample parameters for the betters and create a list of betters
    return np.random.multivariate_normal(means, cov_matrix, N)
