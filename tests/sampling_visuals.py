import numpy as np
import matplotlib.pyplot as plt

def estimate_empirical_covariance(samp, do_variance=True):
    """ Use the unbiased empirical sample covariance as the estimator of the true covariance
    between each pair of variables (columns) in samp, and build the covariance matrix from them.

    Args:
        samp (np.array): nxp matrix for n samples of p dimensions each.
            Pass the values of a dataframe for proper slicing.
        do_variance (bool): if False, do no compute the variance of the estimator.
            An array of zeros is returned instead (for consistency of returns).
            Useful for computing bootstrap replicates, for instance.
            Default: True.

    Returns:
        cov (np.array): the estimated covariance matrix, shape pxp.
        variances (np.array): the variance on each entry of cov, shape pxp.
    """
    p = samp.shape[1]  # Number of variables, p
    N = samp.shape[0]  # Number of points
    cov = np.zeros([p, p])

    # Compute useful moments and central moments of each variable
    # Avoids computing them many times,
    # and low memory requirement O(p) (versus O(np) for samp)
    m1 = np.mean(samp, axis=0)  # first moment, \bar{x}
    m2 = np.mean(samp**2, axis=0)  # \bar{x^2}

    # First, compute the diagonal terms: usual variances
    # Can extract all diagonal terms to 1D array and use element-wise
    cov[np.diag_indices(p)] = N / (N - 1) * (m2 - m1**2)

    # Second, compute the estimate of covariances (upper triangular part of the symmetric matrix)
    for i in range(p):
        x = samp[:, i]
        for j in range(i+1, p):
            y = samp[:, j]
            vxy = N / (N - 1) * (np.mean(x*y) - m1[i]*m1[j])
            cov[i, j] = vxy
            cov[j, i] = vxy  # fill the other half of the symmetric matrix

    # Third, compute the variance of the diagonal terms
    variances = np.zeros([p, p])
    if not do_variance:
        return cov, variances

    # Else: no need for the indent because we returned already
    # 1/N(n_4 - (N-3)/(N-1) (sigma^2)^2)
    n4 = np.mean((samp - m1[None, :])**4, axis=0)  # \frac1N \sum_i (x_i - \bar{x})^4
    variances[np.diag_indices(p)] = (n4 - (N-3)/(N-1) * cov[np.diag_indices(p)]**2) / N

    # Fourth, compute the variance of off-diagonal terms.
    for i in range(p):
        x = samp[:, i]
        for j in range(i+1, p):
            y = samp[:, j]
            x2y2 = np.mean(x**2 * y**2)
            x2y = np.mean(x**2 * y)
            xy2 = np.mean(x * y**2)
            xy = np.mean(x * y)
            nn1 = N * (N - 1)
            varvxy = x2y2 / N - (N-2)/nn1 * xy**2 + m2[i]*m2[j]/nn1
            varvxy += 2*(3*N - 4)/nn1 * xy * m1[i] * m1[j] - 2 * (2*N - 3)/nn1 * m1[i]**2 * m1[j]**2
            varvxy -= 2*(N-2)/nn1 * (x2y * m1[j] + xy2 * m1[i])
            varvxy += (N-4)/nn1 * (m2[i] * m1[j]**2 + m1[i]**2 * m2[j])
            variances[i, j] = varvxy
            variances[j, i] = varvxy

    return cov, variances

if __name__ == "__main__":
    test_type = 3  # 1 (normal sampling) or 2 (multi sampling) or 3 (multi pdf)
    if test_type == 1:
        import seaborn as sns
        arra = np.fromfile("1d_normal_samples.bin")
        x1 = arra[::2]
        x2 = arra[1::2]
        #plt.scatter(x1, x2, s=1)
        #plt.axis("equal")
        #plt.show()
        #plt.close()

        # Histogram, should look like normal(0, 1)
        xrange = np.arange(-6, 6, 0.1)
        ax = sns.kdeplot(data=arra)
        normal_dist = np.exp(-xrange**2 / 2) / np.sqrt(2*np.pi)
        ax.plot(xrange, normal_dist, label="N(0, 1) ref")
        ax.set_yscale("log")
        ax.legend()
        plt.show()
        plt.close()

    ## 2D multinormal sampling
    elif test_type == 2:
        # Covariance chosen: {{1., -0.5}, {-0.5, 1.}}
        # Means: {3., -3.}
        # 10000 samples
        fig, ax = plt.subplots()
        arra = np.fromfile("2d_multinormal_samples.bin").reshape(-1, 2)
        ax.scatter(arra[:, 0], arra[:, 1], s=2)
        # ax.legend()
        plt.show()
        plt.close()

        # Try recovering the covariance matrix and the mean
        covar, varierr = estimate_empirical_covariance(arra, do_variance=True)
        means = np.mean(arra, axis=0)
        print("Fitted covariance matrix on the generated multinormal")
        print(covar)
        print("Error on those estimates:\n", varierr)
        print("Fitted means:\n", means)

    elif test_type == 3:
        import scipy as sp
        from scipy import stats
        # Covariance chosen: {{1., -0.5}, {-0.5, 1.}}
        # Means: {3., -3.}
        # Grid: x in [-10, 10] left to right, y is [10, -10] top to bottom
        # step size 0.1, so 201 elements in each direction.
        means = np.array([3., -3])
        covmat = np.array([[1., -0.5], [-0.5, 1.]])
        rv = sp.stats.multivariate_normal(means, covmat)
        xgrid, ygrid = np.meshgrid(np.linspace(-10, 10, 401),
                                    np.linspace(10, -10, 401), indexing="xy")
        pos = np.dstack((xgrid, ygrid))
        fx_py = rv.pdf(pos)

        # Import the C output also
        fx_c = np.fromfile("multinormal_pdf.bin").reshape(-1, 401)
        diff = (fx_py - fx_c)/fx_py

        # Compare on a plot
        fig, ax = plt.subplots()
        im = ax.imshow(diff, cmap="RdBu", extent=(-10., 10.)*2)
        fig.colorbar(im, ax=ax, label="Relative error")
        plt.show()
        plt.close()
