import numpy as np
import scipy.sparse.linalg as linalg


def get_pagerank(transition_prob_matrix, alpha=0.85):
    n = len(transition_prob_matrix)
    transition_prob_matrix += np.ones([n,n])*alpha/n
    lamb, v = linalg.eigs(transition_prob_matrix, k=1)
    P = v[:, 0]
    P /= P.sum()
    return np.argsort(P)[::-1], P


def get_easy_pagerank(transition_prob_matrix, err_dist=0.0001, alpha=0.85):
    n = len(transition_prob_matrix)
    transition_prob_matrix += np.ones([n,n])*alpha/n
    P = np.array([1/n]*n).reshape(n, 1)
    while True:
        prev = P.copy()
        P = transition_prob_matrix.dot(P)
        P /= P.sum()
        err = np.abs(P-prev).sum()
        if err <= err_dist:
            break
        transition_prob_matrix = transition_prob_matrix.dot(transition_prob_matrix)
    P = np.array(P.T)[0]
    return np.argsort(P)[::-1], P


if __name__ == '__main__':
    M = np.array([[0, 1, 1. / 2, 0, 1. / 4, 1. / 2, 0],
                  [1. / 5, 0, 1. / 2, 1. / 3, 0, 0, 0],
                  [1. / 5, 0, 0, 1. / 3, 1. / 4, 0, 0],
                  [1. / 5, 0, 0, 0, 1. / 4, 0, 0],
                  [1. / 5, 0, 0, 1. / 3, 0, 1. / 2, 1],
                  [0, 0, 0, 0, 1. / 4, 0, 0],
                  [1. / 5, 0, 0, 0, 0, 0, 0]])
    print("get pagerank------------------")
    print("Rank | ID | Prob")
    r,  p = get_pagerank(M, alpha=0)
    for i in range(len(r)):
        print(f'{i+1} | {r[i]+1} | {p[r[i]]}')

    print("get easy pagerank-------------")
    print("Rank | ID | Prob")
    r,  p = get_easy_pagerank(M, alpha=0)
    for i in range(len(r)):
        print(f'{i+1} | {r[i]+1} | {p[r[i]]}')