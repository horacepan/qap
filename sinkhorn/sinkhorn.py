import pdb
import numpy as np

def sinkhorn(mat):
    rows, cols = mat.shape
    row_normalized = mat / mat.sum(1).reshape((rows, 1))
    col_normalized = row_normalized / row_normalized.sum(0).reshape((1, cols))
    return col_normalized

def check_doubly_stoch(mat, tol):
    r, c = mat.shape
    row_sums = mat.sum(1)
    col_sums = mat.sum(0)

    return np.allclose(row_sums, np.ones(r), atol=tol) and \
           np.allclose(col_sums, np.ones(c), atol=tol)

def row_sum_err(mat):
    rows, cols = mat.shape
    row_sum = mat.sum(1)
    col_sum = mat.sum(0)
    # get euclidean dist to 1

    row_sq_diff = np.sum(np.square(row_sum - np.ones(rows)))
    col_sq_diff = np.sum(np.square(col_sum - np.ones(cols)))
    return row_sq_diff + col_sq_diff

def test(n, tol):
    mat = np.abs(np.random.random((n, n)))
    i = 0
    while row_sum_err(mat) > tol:
        print('Iter {} | Err: {}'.format(i, row_sum_err(mat)))
        mat = sinkhorn(mat)
        i += 1

    return mat

if __name__ == '__main__':
    n = 100
    m = test(n, 1e-6)

    print("Row sum")
    print(m.sum(1))
    print("Col sum")
    print(m.sum(0))
    pdb.set_trace()
