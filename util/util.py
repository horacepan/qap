import pdb
import sys
import numpy as np

sys.path.append('../')
DATDIR = 'qapdata'
SLNDIR = 'qapsoln'

def make_perm(perm_lst):
    '''
    Maps the permutation list(where the permutation is defined as list index -> perm(list index)
    IE: [1, 3, 2] is the permutation (2 3)
        [3, 1, 2] is the permutation (1 3 2)
    Returns a permutation function.
    IE: make_perm([3, 1, 2]) returns a function f such that f(1) = 3, f(2) = 1, f(3) = 2
    '''
    # convert to dict. List is 0-indexed but the perm we want is 1 to n
    perm_dict = {i+1: pi for i, pi in enumerate(perm_lst)}
    def perm_func(i):
        assert 1 <= i <= len(perm_lst), 'Input to permutation {} is not in the range [1, {}]'.format(i, len(perm_lst))
        return perm_dict[i]

    return perm_func

def make_perm_matrix(perm_lst):
    n = len(perm_lst)
    pmat = np.zeros((n, n))
    for i, _pi in enumerate(perm_lst):
        pi = _pi - 1
        pmat[pi, i] = 1
    return pmat

def make_inv_perm(perm_lst):
    perm_dict = {pi: i+1 for i, pi in enumerate(perm_lst)}
    def inv_perm(i):
        assert 1 <= i <= len(perm_lst)
        return perm_dict[i]
    return inv_perm

def apply_perm(B, perm):
    out = np.zeros(B.shape)
    for i in range(len(B)):
        for j in range(len(B)):
            p_i = perm(i+1) - 1
            p_j = perm(j+1) - 1
            out[p_i, p_j] = B[i, j]

    # another way

    return out
def read_problem(fname):
    with open(fname, 'r') as f:
        n = int(f.readline().strip())
        A = np.zeros((n, n))
        B = np.zeros((n, n))

        # skip line
        f.readline()
        for i in range(n):
            A[i, :] = list(map(int, f.readline().strip().split()))

        # skip line
        f.readline()
        for i in range(n):
            B[i, :] = list(map(int, f.readline().strip().split()))

        return A, B

def read_soln(fname):
    with open(fname, 'r') as f:
        n, val = map(int, f.readline().strip().split())
        perm_lst = list(map(int, f.readline().strip().split()))
        perm = make_perm(perm_lst)
        return val, perm, perm_lst

def check_soln(A, B, perm, soln):
    '''
    A: numpy square matrix
    B: numpy square matrix of the same shape as A
    perm: a function from [1, len(A)] -> [1, len(A)]
    soln: int of the solution to the qap problem on A, B
    '''
    assert A.shape == B.shape
    n = len(A)
    total = 0

    for i in range(n):
        for j in range(n):
            # i, j are 0 indexed but the permutations are functions from [1, n] -> [1, n]
            p_i = perm(i+1) - 1
            p_j = perm(j+1) - 1
            total += A[i, j] * B[p_i, p_j]
    assert total == soln

def check_instance(name):
    '''
    name is the string name of the qap problem instance.
    Ex: nug12, nug25, etc
    '''
    dat_name = '../data/{}/{}.dat'.format(DATDIR, name)
    sln_name = '../data/{}/{}.sln'.format(SLNDIR, name)
    A, B = read_problem(dat_name)
    val, perm, perm_lst = read_soln(sln_name)
    check_soln(A, B, perm, val)

    # another check via creating the perm matrix
    inv_perm = make_inv_perm(perm_lst)
    perm_B = apply_perm(B, inv_perm)
    sol = np.sum(np.multiply(A, perm_B))

    sol2 = 0
    for i in range(len(A)):
        for j in range(len(A)):
            sol2 += A[i, j] * perm_B[i, j]
    if sol != val:
        pdb.set_trace()
    if sol2 != val:
        pdb.set_trace()
    print('Instance {} is okay'.format(name))

if __name__ == '__main__':
    pmat = make_perm_matrix([2, 1, 3])
    pdb.set_trace()
    check_instance('nug12')
