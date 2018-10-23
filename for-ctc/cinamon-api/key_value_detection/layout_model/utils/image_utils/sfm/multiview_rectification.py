from __future__ import unicode_literals
from __future__ import print_function
from numpy import zeros, matmul, diag


def get_A_b(N, I, W, permutation_mat, lower_A, observed_z):
    A = zeros((N+2*I, I))
    A[0:N, :] = matmul(W, permutation_mat)
    A[N:, :] = lower_A
    b = zeros((N+2*I, 1))
    b[0:N, :] = matmul(W, observed_z)
    return A, b


def update_W(W, permutation_mat, z_old, observed_z, eps=10**-8):
    r = matmul(matmul(W, permutation_mat), z_old) - matmul(W, observed_z)
    R = diag(r) + eps
    return 1.0/R
