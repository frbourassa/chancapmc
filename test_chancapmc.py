import chancapmc
import numpy as np
import os
from itertools import product

def test_memory():
    print("PID:", os.getpid())
    input()
    print("Created arrays in Python")
    n = 6
    a1 = np.arange(n**2, dtype=float).reshape([n]*2) * 1.3123
    a2 = np.arange(n**3, dtype=float).reshape([n]*3)*1.5
    a3 = np.arange(n*2, dtype=np.float64) + 1
    a3 = a3[::2]  # Check what happens when we use a non-contiguous view, does GETPTR still work?
    for i in range(4):
        input()
        print("Launching chancapmc.ba_discretein_gaussout for a memory test")
        chancapmc.ba_discretein_gaussout(a1, a2, a3, 20.)  # captol > 10. for memory test
    print("Final hold")
    input()
    return None

def test_blahut():
    print("PID:", os.getpid())
    input()
    # Basically run the same two blahut-arimoto tests as in unittests.c
    # but initialized from the Python interface.
    # First, for a binary channel, two well-separated, symmetric gaussians
    means = np.asarray([[6, 6], [-6, -6]], dtype=np.float64)
    covs = np.asarray([[[1, 0], [0, 1]]]*2, dtype=np.float64)
    inputs = np.asarray([1., 2.])
    sd = 946373  # seed
    cap_and_vec = chancapmc.ba_discretein_gaussout(means, covs, inputs, 5., sd)
    print("Capacity found, for 2 gaussians:", cap_and_vec[0], "bits")
    print("Optimal probability vector:", cap_and_vec[1])
    print()

    # Second, four symmetric gaussians with some overlap
    means = np.asarray([[1, 1], [1, -1], [-1, 1], [-1, -1]], dtype=np.float64)
    covs = np.asarray([[[1, 0], [0, 1]]]*4, dtype=np.float64)
    inputs = np.asarray([1., 2., 3., 4.])
    sd = 946373
    cap_and_vec = chancapmc.ba_discretein_gaussout(means, covs, inputs, 5., sd)
    print("Capacity found, for 4 gaussians:", cap_and_vec[0], "bits")
    print("Optimal probability vector:", cap_and_vec[1])


def test_blahut_grabowski():
    """ Code adapted from test cases in
    https://github.com/pawel-czyz/channel-capacity-estimator
    """
    meanvecs = np.zeros([8, 3])
    covvecs = np.zeros([8, 3, 3])
    corners = list(product(range(2),repeat=3))
    for i in range(8):
        means3, sigma = (corners[i], 0.25 + ((i + 1)/8 + sum(corners[i]))/10)
        covar3 = sigma**2 * np.identity(3)
        meanvecs[i] = np.asarray(means3)
        covvecs[i] = np.asarray(covar3)
    print(meanvecs)
    print(covvecs)
    cap_accur = 1.8035  # result from Mathematica, according to Grabowski 2019
    rtol = 0.01
    sd = 6823949
    res = chancapmc.ba_discretein_gaussout(meanvecs, covvecs, np.arange(8), rtol, sd)
    assert abs(res[0] - cap_accur)/cap_accur <= rtol, "Could not reproduce cap. for 8 gaussians on corners of a box"
    print("Capacity for 8 gaussians on corners of a 3D box:", res[0])
    print("Optimal input prob. distrib.:", res[1])

def gaussxw(n):
    """ Fonction qui calcule les poids w_k et les positions x_k à utiliser pour la quadrature gaussienne à n points.
    Les valeurs retournées sont normalisées pour un intervalle d'intégration [-1, 1].
    Les positions x_k sont les zéros du polynôme de Legendre de degré n
    Les poids sont donnés par:
        (\frac{2}{1-x^2} (\frac{dP_n}{dx})^{-2})_{x = x_k}

    Pour une intégrale sur l'intervalle [a, b], les bons x_k' et w_k' s'obtiennent à partir des valeurs pour [-1, 1]
    avec la relation linéaire:
        x_k' = 1/2 (b - a) x_k + 1/2 (b + a)
        w_k' = 1/2 (b - a) w_k

    Args:
        n (int): le nombre de points à utiliser pour la quadrature gaussienne

    Returns:
        x (list): la liste des positions x_k pour [-1, 1]
        w (list): la liste des poids w_k pour chaque x_k de [-1, 1]
    """
    # Estimés des zéros initiaux (doivent être assez bons pour que la méthode converge)
    # Pour ces estimés, formule de Abramowitz et Stegun:
    # x_k = cos(pi*a_k + 1/(8n^2 tan(a_k)) avec a_x = (4k-1)/(4n+2), k=1, 2, 3, ..., n
    a = np.linspace(3, 4*n - 1, n)/(4*n + 2)
    x = np.cos(np.pi*a + 1/(8*n*n*np.tan(a)))

    # Trouver les zéros avec la méthode de Newton-Raphson
    epsilon = 1e-15
    delta = 1.0
    while delta > epsilon:
        # Polynôme P_0(x) = 1 partout
        p0 = np.ones(n, dtype=float)
        # Polynôme P_1(x) = x, on l'évalue aux estimés actuels
        p1 = x.copy()

        # Calculer P_n(x) à chaque zéro
        # On utilise la formule de récurrence de Bonnet, (m+1)P_{m+1}(x) = (2m+1)x P_m(x) - m P_{m-1}(x)
        for m in range(1, n):
           p0, p1 = p1, ((2*m + 1)*x*p1 - m*p0)/(m + 1)

        # Calcul de la correction à apporter aux estimés actuels, soit P_n(x)/P_n'(x)
        # Pour calculer la dérivée de P_n(x), on utilise la relation de récurrence:
        # (x^2 - 1)/n dP_n(x)/dx = xP_n(x) - P_{n-1}(x)
        derivp = (n + 1)*(p0 - x*p1)/(1 - x*x)  # La bonne formule aurait n et non n+1 comme facteur,
                #  mais Newman utilise ceci, j'ignore pourquoi
        varix = p1/derivp
        x -= varix

        # Calcul de la variation
        delta = np.max(abs(varix))

    # Calcul des poids avec ces zéros. Il faut ici annuler le n+1 mis à la place de n plus tôt dans derivp
    w = 2*(n + 1)*(n + 1) / ((1 - x*x) * n*n*derivp*derivp)

    return x, w

def test_blahut_smallnoise():
    """ Test case I computed analytically myself in a small-noise approx.
    """
    import matplotlib.pyplot as plt
    nvecs = 32
    alph = 2**(-8)
    qrange = np.linspace(0., 1., nvecs+1)
    qrange = (qrange[1:] + qrange[:-1])/2
    meanvecs = np.ones([nvecs, 2])*qrange.reshape(nvecs, -1) / np.sqrt(2)
    sigma_mat = np.asarray([[5/8, -3/8], [-3/8, 5/8]])
    covmats = alph * np.tile(sigma_mat, [nvecs, 1, 1]) * (qrange.reshape(nvecs, 1, 1) + 1)**2

    cap_accur = -np.log2(alph) - np.log2(2 * np.pi * np.e)

    rtol = 0.025
    sd = 6823949
    new = True
    if new:
        res = chancapmc.ba_discretein_gaussout(meanvecs, covmats, qrange, rtol, sd)
        np.save("tests/optimal_distrib.npy", res[1])
        print("Capacity found numerically, small noise approx, bits:", res[0])
        # assert abs(res[0] - cap_accur)/cap_accur <= rtol*5, "Could not reproduce cap. for small noise approx."
    else:
        res = [1.6, 0]
        res[1] = np.load("tests/optimal_distrib.npy")
        print(res[1])

    # Plot the prob distrib
    fig, ax = plt.subplots()
    width = np.concatenate([[0.], (qrange[1:] + qrange[:-1])/2, [1.]])
    width = width[1:] - width[:-1]
    ax.bar(qrange, res[1] / width, width=width)
    # Compare to the theoretical one
    theoretical = 2 / (1 + qrange)**2
    ax.plot(qrange, theoretical, color="orange")
    plt.show()
    plt.close()


if __name__ == "__main__":
    # test_memory()
    # test_blahut()
    # test_blahut_grabowski()
    test_blahut_smallnoise()
