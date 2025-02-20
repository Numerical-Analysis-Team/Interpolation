
def lagrange_interpolation(table, x):
    X, Y = table
    n = len(X)
    result = 0
    for i in range(n):
        term = Y[i]
        for j in range(n):
            if i != j:
                term *= (x - X[j]) / (X[i] - X[j])
        result += term
    return result


def neville_interpolation(table, x):
    X, Y = table
    n = len(X)
    Q = [[0 for _ in range(n)] for _ in range(n)]


    for i in range(n):
        Q[i][0] = Y[i]


    for j in range(1, n):
        for i in range(n - j):
            Q[i][j] = ((x - X[i + j]) * Q[i][j - 1] + (X[i] - x) * Q[i + 1][j - 1]) / (X[i] - X[i + j])

    return Q[0][n - 1]


def main():

    X = [1, 2, 3, 4]  # X values
    Y = [1, 4, 9, 16]  # Y values (e.g., y = x^2)

    # The point we want to interpolate
    x = 2.5


    lagrange_result = lagrange_interpolation((X, Y), x)
    neville_result = neville_interpolation((X, Y), x)

    # Print the results
    print(f"Interpolation result using Lagrange method when point x = {x}: {lagrange_result}")
    print(f"Interpolation result using Neville method when point x = {x}: {neville_result}")


if __name__ == "__main__":
    main()
