from Matrix_Unity import *

def linearInterpolation(dataPts, valToInterpolate):
    """
    Linear interpolation or extrapolation for a given point
    """
    xVals = []
    approximatedRes = 0
    flagCheck = 1
    for i in range(len(dataPts)):
        xVals.append(dataPts[i][0])
    for i in range(len(xVals) - 1):
        if i <= valToInterpolate <= i + 1:
            x1 = dataPts[i][0]
            x2 = dataPts[i + 1][0]
            y1 = dataPts[i][1]
            y2 = dataPts[i + 1][1]
            approximatedRes = (((y1 - y2) / (x1 - x2)) * valToInterpolate) + ((y2 * x1) - (y1 * x2)) / (x1 - x2)
            print("\nThe approximation (interpolation) of the point", valToInterpolate, "is:", round(approximatedRes, 4))
            flagCheck = 0
    if flagCheck:
        x1 = dataPts[0][0]
        x2 = dataPts[1][0]
        y1 = dataPts[0][1]
        y2 = dataPts[1][1]
        slope = (y1 - y2) / (x1 - x2)
        approximatedRes = y1 + slope * (valToInterpolate - x1)
        print("\nThe approximation (extrapolation) of the point", valToInterpolate, "is:", round(approximatedRes, 4))
    return approximatedRes

def solveGaussJordan(mtx, vec):
    """
    Solve linear system Ax = b using matrix inversion (Gauss-Jordan)
    """
    mtx, vec = swapPivotRow(mtx, vec)
    invM = invertMatrix(mtx, vec)
    return matVectorMult(invM, vec)

def deriveUMatrix(mtx, vec):
    """
    Break matrix into upper triangular (U)
    """
    Ures = buildIdentityMat(len(mtx), len(mtx))
    for i in range(len(mtx[0])):
        mtx, vec = exchangeRowIfPivotZero(mtx, vec)
        for j in range(i + 1, len(mtx)):
            eMat = buildIdentityMat(len(mtx[0]), len(mtx))
            eMat[j][i] = -(mtx[j][i]) / mtx[i][i]
            mtx = matMult(eMat, mtx)
    Ures = matMult(Ures, mtx)
    return Ures

def deriveLMatrix(mtx, vec):
    """
    Break matrix into lower triangular (L)
    """
    Lres = buildIdentityMat(len(mtx), len(mtx))
    for i in range(len(mtx[0])):
        mtx, vec = exchangeRowIfPivotZero(mtx, vec)
        for j in range(i + 1, len(mtx)):
            eMat = buildIdentityMat(len(mtx[0]), len(mtx))
            eMat[j][i] = -(mtx[j][i]) / mtx[i][i]
            Lres[j][i] = (mtx[j][i]) / mtx[i][i]
            mtx = matMult(eMat, mtx)
    return Lres

def solveLU(mtx, vec):
    """
    Solve linear system Ax = b via LU decomposition
    """
    Umat = deriveUMatrix(mtx, vec)
    Lmat = deriveLMatrix(mtx, vec)
    return matMult(invertMatrix(Umat), matMult(invertMatrix(Lmat), vec))

def solveMatrix(mtxA, vecB):
    """
    General solver that checks determinant.
    If det != 0 => solve with Gauss-Jordan,
    otherwise do LU decomposition.
    """
    detA = calcDeterminant(mtxA, 1)
    print("\nDET(A) =", detA)
    if detA != 0:
        print("\nNon-Singular Matrix - Perform Gauss-Jordan")
        solution = solveGaussJordan(mtxA, vecB)
        print(solution)
        return solution
    else:
        print("Singular Matrix - Perform LU Decomposition\n")
        print("Matrix U:\n", deriveUMatrix(mtxA, vecB))
        print("\nMatrix L:\n", deriveLMatrix(mtxA, vecB))
        print("\nMatrix A=LU:\n", matMult(deriveLMatrix(mtxA, vecB), deriveUMatrix(mtxA, vecB)))
        return matMult(deriveLMatrix(mtxA, vecB), deriveUMatrix(mtxA, vecB))

def polynomialInterpolation(dataPts, xVal):
    """
    Polynomial interpolation function for set of points
    """
    polyMat = [[point[0] ** i for i in range(len(dataPts))] for point in dataPts]
    bVec = [[point[1]] for point in dataPts]

    print("The matrix obtained from the points:\n", polyMat)
    print("\nb vector:\n", bVec)

    matrixSolution = solveMatrix(polyMat, bVec)
    interpolationRes = sum([matrixSolution[i][0] * (xVal ** i) for i in range(len(matrixSolution))])

    print("\nThe polynomial:")
    terms = []
    for i in range(len(matrixSolution)):
        coeff = matrixSolution[i][0]
        terms.append(f"({coeff}) * x^{i}")
    print('P(X) = ' + ' + '.join(terms))
    print(f"\nThe result of P(X={xVal}) is:", interpolationRes)
    return interpolationRes

if __name__ == '__main__':
    dataPoints = [(0, 0), (1, 0.8415), (2, 0.9093), (3, 0.1411), (4, -0.7568), (5, -0.9589), (6, -0.2794)]
    xTarget = 1.28

    print("----------------- Interpolation & Extrapolation Methods -----------------\n")
    print("Table Points:")
    for pt in dataPoints:
        print(f"({pt[0]}, {pt[1]})")
    print("\nFinding an approximation to the point:", xTarget)

    print("\n--- Linear Interpolation ---")
    linearRes = linearInterpolation(dataPoints, xTarget)
    print(f"Linear Interpolation Result: {linearRes}")

    print("\n--- Polynomial Interpolation ---")
    polyRes = polynomialInterpolation(dataPoints, xTarget)
    print(f"Polynomial Interpolation Result: {polyRes}")

    print("\n---------------------------------------------------------------------------\n")
