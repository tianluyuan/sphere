import numpy as np

import scipy.linalg.lapack
import scipy.linalg.decomp

def matrix_balance(A, permute=True, scale=True, separate=False,
	overwrite_a=False):
    """
    Compute a diagonal similarity transformation for row/column balancing.
    The balancing tries to equalize the row and column 1-norms by applying
    a similarity transformation such that the magnitude variation of the
    matrix entries is reflected to the scaling matrices.
    Moreover, if enabled, the matrix is first permuted to isolate the upper
    triangular parts of the matrix and, again if scaling is also enabled,
    only the remaining subblocks are subjected to scaling.
    The balanced matrix satisfies the following equality
    .. math::
			B = T^{-1} A T
    The scaling coefficients are approximated to the nearest power of 2
    to avoid round-off errors.
    Parameters
    ----------
    A : (n, n) array_like
	Square data matrix for the balancing.
    permute : bool, optional
	The selector to define whether permutation of A is also performed
	prior to scaling.
    scale : bool, optional
	The selector to turn on and off the scaling. If False, the matrix
	will not be scaled.
    separate : bool, optional
	This switches from returning a full matrix of the transformation
	to a tuple of two separate 1D permutation and scaling arrays.
    overwrite_a : bool, optional
	This is passed to xGEBAL directly. Essentially, overwrites the result
	to the data. It might increase the space efficiency. See LAPACK manual
	for details. This is False by default.
    Returns
    -------
    B : (n, n) ndarray
	Balanced matrix
    T : (n, n) ndarray
	A possibly permuted diagonal matrix whose nonzero entries are
	integer powers of 2 to avoid numerical truncation errors.
    scale, perm : (n,) ndarray
	If ``separate`` keyword is set to True then instead of the array
	``T`` above, the scaling and the permutation vectors are given
	separately as a tuple without allocating the full array ``T``.
    Notes
    -----
    This algorithm is particularly useful for eigenvalue and matrix
    decompositions and in many cases it is already called by various
    LAPACK routines.
    The algorithm is based on the well-known technique of [1]_ and has
    been modified to account for special cases. See [2]_ for details
    which have been implemented since LAPACK v3.5.0. Before this version
    there are corner cases where balancing can actually worsen the
    conditioning. See [3]_ for such examples.
    The code is a wrapper around LAPACK's xGEBAL routine family for matrix
    balancing.
    .. versionadded:: 0.19.0
    Examples
    --------
    >>> from scipy import linalg
    >>> x = np.array([[1,2,0], [9,1,0.01], [1,2,10*np.pi]])
    >>> y, permscale = linalg.matrix_balance(x)
    >>> np.abs(x).sum(axis=0) / np.abs(x).sum(axis=1)
    array([ 3.66666667,  0.4995005 ,  0.91312162])
    >>> np.abs(y).sum(axis=0) / np.abs(y).sum(axis=1)
    array([ 1.2       ,  1.27041742,  0.92658316])  # may vary
    >>> permscale  # only powers of 2 (0.5 == 2^(-1))
    array([[  0.5,   0. ,  0. ],  # may vary
	[  0. ,   1. ,  0. ],
	[  0. ,   0. ,  1. ]])
    References
    ----------
    .. [1] : B.N. Parlett and C. Reinsch, "Balancing a Matrix for
       Calculation of Eigenvalues and Eigenvectors", Numerische Mathematik,
       Vol.13(4), 1969, DOI:10.1007/BF02165404
    .. [2] : R. James, J. Langou, B.R. Lowery, "On matrix balancing and
       eigenvector computation", 2014, Available online:
	   https://arxiv.org/abs/1401.5766
    .. [3] :  D.S. Watkins. A case where balancing is harmful.
       Electron. Trans. Numer. Anal, Vol.23, 2006.
    """

    A = np.atleast_2d(scipy.linalg.decomp._asarray_validated(A, check_finite=True))

    if not np.equal(*A.shape):
	raise ValueError('The data matrix for balancing should be square.')

    gebal = scipy.linalg.lapack.get_lapack_funcs(('gebal'), (A,))
    B, lo, hi, ps, info = gebal(A, scale=scale, permute=permute,
	    overwrite_a=overwrite_a)

    if info < 0:
	raise ValueError('xGEBAL exited with the internal error '
		'"illegal value in argument number {}.". See '
		'LAPACK documentation for the xGEBAL error codes.'
		''.format(-info))

	# Separate the permutations from the scalings and then convert to int
    scaling = np.ones_like(ps, dtype=float)
    scaling[lo:hi+1] = ps[lo:hi+1]

    # gebal uses 1-indexing
    ps = ps.astype(int, copy=False) - 1
    n = A.shape[0]
    perm = np.arange(n)

    # LAPACK permutes with the ordering n --> hi, then 0--> lo
    if hi < n:
	for ind, x in enumerate(ps[hi+1:][::-1], 1):
	    if n-ind == x:
		continue
	    perm[[x, n-ind]] = perm[[n-ind, x]]

    if lo > 0:
	for ind, x in enumerate(ps[:lo]):
	    if ind == x:
		continue
	    perm[[x, ind]] = perm[[ind, x]]

    if separate:
	return B, (scaling, perm), lo, hi

    # get the inverse permutation
    iperm = np.empty_like(perm)
    iperm[perm] = np.arange(n)

    return B, np.diag(scaling)[iperm, :], lo, hi

##' Calculation of e^A with the Scaling & Squaring Method with Balancing
##' according to Higham (2008)
##'
##' R-Implementation of Higham's Algorithm from the Book (2008)
##' "Functions of Matrices - Theory and Computation", Chapter 10, Algorithm 10.20
##' Step 0:    Balancing
##' Step 1:    Scaling
##' Step 2:    Pade-Approximation
##' Step 3:    Squaring
##' Step 4:    Reverse Balancing
##'
##' @title Matrix Exponential with Scaling & Squaring and Balancing
##' @param A nxn Matrix
##' @param balancing logical indicating if balancing (step 0) should be applied
##' @return e^A Matrixeponential; nxn Matrix
##' @author Martin Maechler
def expm_Higham08(A, balancing=True):
    A = np.asarray(A)
    # Check if A is square
    d = A.shape
    if len(d) != 2 or d[0] != d[1]:
	raise ValueError("'A' must be a square matrix")
    n = d[0]

    if n <= 1:
	return np.exp(A)

    ## else  n >= 2 ... non-trivial case : -------------

    ##---------STEP 0: BALANCING------------------------------------------------
    ## if balancing is asked for, balance the matrix A

    if balancing:
	baP_B, (baP_scale, baP_perm), baP_lo, baP_hi = matrix_balance(A, separate=True, scale=False, permute=True)
	baS_B, (baS_scale, baS_perm), baS_lo, baS_hi = matrix_balance(A, separate=True, scale=True, permute=False)
	A = np.copy(baS_B)

    ##--------STEP 1 and STEP 2 SCALING & PADE APPROXIMATION--------------------

    ## Informations about the given matrix
    nA = scipy.linalg.norm(A, ord=1)

    ## try to remain in the same matrix class system:
    I = np.identity(n)

    ## If the norm is small enough, use the Pade-Approximation (PA) directly
    if nA <= 2.1:
	t = np.array([0.015, 0.25, 0.95, 2.1])
	    ## the minimal m for the PA :
	l = np.argmax(nA <= t)

	## Calculate PA
	C = np.array(((120,60,12,1,0,0,0,0,0,0), (30240,15120,3360,420,30,1,0,0,0,0), (17297280,8648640,1995840,277200,25200,1512,56,1,0,0), (17643225600,8821612800,2075673600,302702400,30270240, 2162160,110880,3960,90,1)))
	A2 = A.dot(A)
	P = np.copy(I)
	U = C[l,1]*I
	V = C[l,0]*I

	for k in range(l):
	    P = P.dot(A2)
	    U = U + C[l,(2*k)+1]*P
	    V = V + C[l,(2*k)]*P
	U = A.dot(U)
	X = scipy.linalg.solve(V-U,V+U)

    ## Else, check if norm of A is small enough for m=13.
    ## If not, scale the matrix
    else:
        s = np.log2(nA/5.4)
	B = np.copy(A)
	## Scaling
	if s > 0:
	    s = int(np.ceil(s))
	    B = B/(2.0**s)

	## Calculate PA

	c2 = np.array([64764752532480000,32382376266240000,7771770303897600, 1187353796428800, 129060195264000,10559470521600, 670442572800, 33522128640, 1323241920, 40840800,960960,16380, 182,1])

	B2 = B.dot(B)
	B4 = B2.dot(B2)
	B6 = B2.dot(B4)

	U = B.dot(B6.dot(c2[13]*B6 + c2[11]*B4 + c2[9]*B2) + c2[7]*B6 + c2[5]*B4 + c2[3]*B2 + c2[1]*I)
	V = B6.dot(c2[12]*B6 + c2[10]*B4 + c2[8]*B2) + c2[6]*B6 + c2[4]*B4 + c2[2]*B2 + c2[0]*I

	X = scipy.linalg.solve(V-U,V+U)

	##---------------STEP 3 SQUARING----------------------------------------------
	if s > 0:
	    for t in range(s):
		X = X.dot(X)

##-----------------STEP 4 REVERSE BALANCING---------------------------------
    if balancing:##	 reverse the balancing
        d = baS_scale
        X = X * (np.tile(d,n) * np.repeat(1.0/d, n)).reshape(tuple(reversed(X.shape))).T

        ## apply inverse permutation (of rows and columns):
        pp = baP_scale.astype(int)
        if baP_lo > 0:             ## The lower part
            for i in range(baP_lo-1, 1): # 'p1' in *reverse* order
                tt = X[:,i]
                X[:,i] = X[:,pp[i]]
                X[:,pp[i]] = tt
                tt = X[i,:]
                X[i,:] = X[pp[i],:]
                X[pp[i],:] = tt

        if baP_hi < n-1:             ## The upper part
            for i in range(baP_hi+1, n): # 'p2' in *forward* order
                ## swap	 i <-> pp[i]   both rows and columns
                tt = X[:,i]
                X[:,i] = X[:,pp[i]]
                X[:,pp[i]] = tt
                tt = X[i,:]
                X[i,:] = X[pp[i],:]
                X[pp[i],:] = tt
    return X



