import numpy as np
from sklearn.linear_model import Ridge

def ridge(X, y, Lambda):
    n, d = X.shape
    I = np.identity(d)
    I[0, 0] = 0

    XTX = X.T @ X
    XTy = X.T @ y
    
    # inv = np.linalg.inv(XTX + Lambda * I)
    # beta = inv @ XTy
    beta = np.linalg.solve(XTX + 2 * n * Lambda * I, XTy)
    return beta

def ten_fold(X, y, Lambdas):
    k_folds = 10
    X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
    n, d = X.shape
    fold_size = n // k_folds
    
    avg_errors = np.zeros_like(Lambdas)
    
    for i, Lambda in enumerate(Lambdas):
        for k in range(k_folds):
            start = k * fold_size
            end = start + fold_size if k < k_folds - 1 else n
            
            X_val = X[start:end]
            y_val = y[start:end]
            
            X_train = np.concatenate((X[:start], X[end:]), axis=0)
            y_train = np.concatenate((y[:start], y[end:]), axis=0)
            
            beta_train = ridge(X_train, y_train, Lambda)
            
            y_pred = X_val @ beta_train
            avg_errors[i] += np.sum((y_val - y_pred)**2)
            
        avg_errors[i] /= n
    
    Lambda = Lambdas[avg_errors.argmin()]
    # print(avg_errors)
    beta = ridge(X, y, Lambda)
    return beta, Lambda

if __name__ == "__main__":
    with open("code/data/train-matrix.txt", 'r') as f:
        lines = f.readlines()
    rows = int(lines[0].strip())
    cols = int(lines[1].strip())

    X_lines = lines[2:2+rows]
    y_lines = lines[2+rows:]
    
    X = np.array([list(map(float, line.strip().split())) for line in X_lines])
    y = np.array([float(line.strip()) for line in y_lines])

    beta, Lambda = ten_fold(X, y, np.array([0.0125, 0.025, 0.05, 0.1, 0.2]))
    print(beta)
    print(Lambda)

    ridge_sklearn = Ridge(alpha=Lambda, solver='cholesky')
    ridge_sklearn.fit(X, y)
    # print(ridge_sklearn.coef_)
    # print(ridge_sklearn.intercept_)

    with open("code/data/test-matrix.txt", 'r') as f:
        lines = f.readlines()
    rows = int(lines[0].strip())
    cols = int(lines[1].strip())

    X_lines = lines[2:2+rows]
    y_lines = lines[2+rows:]
    
    X_test = np.array([list(map(float, line.strip().split())) for line in X_lines])
    y_test = np.array([float(line.strip()) for line in y_lines])

    X_test = np.concatenate((np.ones((X_test.shape[0], 1)), X_test), axis=1)
    y_pred = X_test @ beta
    error = np.sum(np.abs(y_test - y_pred))
    print(error, y_test-y_pred)

    with open("code/data/true-beta.txt", 'r') as f:
        lines = f.readlines()

    true_beta = np.array([float(line.strip()) for line in lines])
    # print(true_beta)
    # print(beta)
    print(np.sum((true_beta - beta)**2))