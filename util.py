import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

def sample_uniform(xmin=1e-6,xmax=1.0,size=3,rng=None):
    if rng is None:
        rng = np.random.default_rng(seed=42)
    samples = rng.uniform(xmin,xmax,size=size-2)
    xs = [samples[i] for i in range(samples.size)]
    xs.append(xmin)
    xs.append(xmax)
    return np.array(xs)


#chatgpgt
def sample_linear_pdf(xmin=1e-6, xmax=1.0, size=1, rng=None):
    """
    Samples from a normalized linear PDF that decreases from f(xmin)=1 to f(xmax)=0.

    Parameters:
        xmin (float): Lower bound of the interval.
        xmax (float): Upper bound of the interval.
        size (int): Number of samples to draw.
        rng (np.random.Generator): Optional RNG for reproducibility. Defaults to seed=42.

    Returns:
        np.ndarray: Samples drawn from the distribution.
    """
    if rng is None:
        rng = np.random.default_rng(seed=42)

    # Normalization constant: integral of (xmax - x) from xmin to xmax
    norm = (xmax - xmin)**2 / 2

    def normalized_pdf(x):
        return (xmax - x) / norm

    max_pdf = normalized_pdf(xmin)  # peak of PDF

    samples = [xmin,xmax]
    while len(samples) < size:
        x = rng.uniform(xmin, xmax)
        y = rng.uniform(0, max_pdf)
        if y < normalized_pdf(x):
            samples.append(x)

    return np.array(samples)


#chatgpt
def chebyshev_nodes(n, xmin, xmax):
    """
    Compute n Chebyshev nodes of the first kind in the interval [xmin, xmax],
    with the nodes exactly hitting xmin and xmax.

    Parameters:
        n     : number of nodes (n >= 2)
        xmin  : lower bound of interval
        xmax  : upper bound of interval

    Returns:
        numpy array of Chebyshev nodes in [xmin, xmax]
    """
    if n < 2:
        raise ValueError("Need at least two nodes to include both endpoints.")

    # Chebyshev nodes on [-1, 1]
    i = np.arange(n)
    x_cheb = np.cos(np.pi * i / (n - 1))  # goes from +1 to -1

    # Map from [-1, 1] to [xmin, xmax]
    x_mapped = 0.5 * (x_cheb + 1) * (xmax - xmin) + xmin


    return x_mapped

def diag(xs):
    m=xs.size
    return sp.diags([xs],[0]).tocsr()


def cg(xs):
    m=xs.size
    A=diag(xs)
    res=[]
    b=np.ones(m)/np.sqrt(m)
    def callback(xk):
        nonlocal res
        res.append(np.linalg.norm(b-A@xk))
    spla.cg(A,b,maxiter=m,rtol=1e-10,callback=callback)
    return res



