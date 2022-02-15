# Python routines to implement various matrix inversion algorithms

import numpy as np


def gaussJordan(A: np.ndarray):
    # Converted pseudocode exmple from Numerical Recipes
    n = A.shape[0]
    idxc, idxr, ipiv = np.zeros(n), np.zeros(n), np.zeros(n)

    for i in range(n):
        big = 0.0
        for j in range(n):
            if ipiv[j] != 1:
                for k in range(n):
                    if ipiv[k] == 0:
                        if abs(A[j, k]) >= big:
                            big = abs(A[j, k])
                            irow, icol = j, k
            ipiv[icol] += 1

            if irow != icol:
                for l in range(n):
                    A[irow, l], A[icol, l] = A[icol, l], A[irow, l]
            idxr[i] = irow
            idxc[i] = icol

            if A[icol, icol] == 0:
                raise ValueError("gaussJordan: Singular Matrix!")

            pivinv = 1.0 / A[icol, icol]
            A[icol, icol] = 1.0

            for l in range(n):
                A[icol, l] *= pivinv

            for ll in range(n):
                if ll != icol:
                    dum = A[ll, icol]
                    A[ll, icol] = 0.0
                    for l in range(n):
                        A[ll, l] -= A[icol, l] * dum

    for l in range(n):
        if idxr[n-1-l] != idxc[n-1-l]:
            for k in range(n):
                A[k, idxr[n-1-l]], A[k, idxc[n-1-l]] = A[k, idxc[n-1-l]], A[k, idxr[n-1-l]]

    return A


def gaussJordanAlt(A: np.ndarray):
    # not in-place version of previous algo, manipulates augmentred matrix
    # Aug = np.concat([A, np.eye(A.shape[0])])
    pass


if __name__ == "__main__":
    A_samp = np.array([[2, 1], [3, 0]])
    A_inv = gaussJordan(A_samp)
