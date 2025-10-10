import numpy as np

def greedy(X, y, K):
    n, d = X.shape
    A = set()
    X = np.concatenate((np.ones((n, 1)), X), axis=1)
    beta = np.zeros(d+1)
    
    for k in range(K + 1):
        residual = y - X @ beta
        corr = np.abs(X.T @ residual)
        for idx in A:
            corr[idx] = -1 
            
        i_k = np.argmax(corr) if k > 0 else 0
        A.add(i_k)
        
        A_list = sorted(list(A))
        X_A = X[:, A_list]
        
        # beta_k = np.linalg.pinv(X_A) @ y
        beta_k = np.linalg.lstsq(X_A, y, rcond=None)[0]
        beta[A_list] = beta_k
            
    return sorted(list(A)[1:]), beta

if __name__ == "__main__":
    with open("code/data/train-matrix.txt", 'r') as f:
        lines = f.readlines()
    rows = int(lines[0].strip())
    cols = int(lines[1].strip())

    X_lines = lines[2:2+rows]
    y_lines = lines[2+rows:]
    
    X = np.array([list(map(float, line.strip().split())) for line in X_lines])
    y = np.array([float(line.strip()) for line in y_lines])

    print(X.shape)
    print(y.shape)

    A_k, beta_k = greedy(X, y, 6)
    print(A_k)
    print(beta_k)