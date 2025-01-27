import numpy as np

def showMatrix(mtx):
    """
    Print matrix in a more readable way
    """
    for rowLine in mtx:
        for val in rowLine:
            print(val, end=" ")
        print()
    print()

def maxRowSumNorm(mtx):
    """
    Calculate the 'max norm' (maximum row sum) of a matrix
    """
    largestRowSum = 0
    for rowIdx in range(len(mtx)):
        tempSum = 0
        for colIdx in range(len(mtx)):
            tempSum += abs(mtx[rowIdx][colIdx])
        if tempSum > largestRowSum:
            largestRowSum = tempSum
    return largestRowSum

def replaceRow(mtx, rowA, rowB):
    """
    Swap two rows within a matrix
    """
    dimension = len(mtx)
    for idx in range(dimension + 1):
        tempVal = mtx[rowA][idx]
        mtx[rowA][idx] = mtx[rowB][idx]
        mtx[rowB][idx] = tempVal

def checkDiagonalDominance(mtx):
    """
    Check if matrix is diagonally dominant
    """
    if mtx is None:
        return False
    mainDiag = np.diag(np.abs(mtx))
    rowSums = np.sum(np.abs(mtx), axis=1) - mainDiag
    return np.all(mainDiag > rowSums)

def checkSquareMat(mtx):
    """
    Check if matrix is square
    """
    if mtx is None:
        return False
    rowCount = len(mtx)
    for rowLine in mtx:
        if len(rowLine) != rowCount:
            return False
    return True

def rearrangeDominantDiagonal(mtx):
    """
    Reorder matrix so that diagonal might be dominant
    """
    n = len(mtx)
    perm = np.argsort(np.diag(mtx))[::-1]
    reorg = mtx[perm][:, perm]
    return reorg

def buildDominantDiagonal(mtx):
    """
    Try to modify matrix to form a dominant diagonal
    """
    sizeN = len(mtx)
    dominants = [0]*sizeN
    diagMat = []
    for rowIdx in range(sizeN):
        for colIdx in range(len(mtx[0])):
            # לבדוק האם הערך גדול מסכום שאר הערכים בשורה
            if (mtx[rowIdx][colIdx] > sum(map(abs,map(int,mtx[rowIdx]))) - mtx[rowIdx][colIdx]):
                dominants[rowIdx] = colIdx

    for rowIdx in range(sizeN):
        diagMat.append([])
        if rowIdx not in dominants:
            print("Couldn't find dominant diagonal.")
            return mtx

    for rowIdx, colIdx in enumerate(dominants):
        diagMat[colIdx] = mtx[rowIdx]
    return diagMat

def elemMatrixSwapRows(n, row1, row2):
    """
    Elementary matrix for swapping between two rows
    """
    eMat = np.identity(n)
    eMat[[row1, row2]] = eMat[[row2, row1]]
    return np.array(eMat)

def elemMatrixProduct(A, B):
    """
    Multiply two matrices (using basic nested loops)
    """
    if len(A[0]) != len(B):
        raise ValueError("Matrix dimensions are incompatible for multiplication.")
    prodMat = [[0 for _ in range(len(B[0]))] for _ in range(len(A))]
    for i in range(len(A)):
        for j in range(len(B[0])):
            for k in range(len(B)):
                prodMat[i][j] += A[i][k] * B[k][j]
    return np.array(prodMat)

def elemMatrixAddRow(n, target, source, coeff=1.0):
    """
    Elementary matrix for row addition: R_target += coeff*R_source
    """
    if target < 0 or source < 0 or target >= n or source >= n:
        raise ValueError("Invalid row indices.")
    if target == source:
        raise ValueError("Source and target rows cannot be the same.")
    eMat = np.identity(n)
    eMat[target, source] = coeff
    return np.array(eMat)

def elemMatrixScalarRow(n, r_idx, multFactor):
    """
    Elementary matrix for scaling a row: R_r_idx *= multFactor
    """
    if r_idx < 0 or r_idx >= n:
        raise ValueError("Invalid row index.")
    if multFactor == 0:
        raise ValueError("Scalar cannot be zero for row multiplication.")
    eMat = np.identity(n)
    eMat[r_idx, r_idx] = multFactor
    return np.array(eMat)

def calcDeterminant(mtx, multiplier):
    """
    Recursive function to compute the determinant of a matrix
    """
    width = len(mtx)
    if width == 1:
        return multiplier * mtx[0][0]
    else:
        sign = -1
        determinantVal = 0
        for idx in range(width):
            subMatrix = []
            for rowLine in range(1, width):
                buff = []
                for colLine in range(width):
                    if colLine != idx:
                        buff.append(mtx[rowLine][colLine])
                subMatrix.append(buff)
            sign *= -1
            determinantVal += multiplier * calcDeterminant(subMatrix, sign * mtx[0][idx])
        return determinantVal

def applyPartialPivoting(A, i, N):
    """
    Perform partial pivoting on matrix A in column i
    """
    pivotRow = i
    maxVal = A[pivotRow][i]
    for rIdx in range(i + 1, N):
        if abs(A[rIdx][i]) > maxVal:
            maxVal = A[rIdx][i]
            pivotRow = rIdx
    if A[i][pivotRow] == 0:
        return "Singular Matrix"
    if pivotRow != i:
        e_matrix = elemMatrixSwapRows(N, i, pivotRow)
        print(f"elementary matrix for swap between row {i} to row {pivotRow} :\n {e_matrix} \n")
        A = np.dot(e_matrix, A)
        print(f"The matrix after elementary operation :\n {A}")
        print("------------------------------------------------------------------")

def matMult(matA, matB):
    """
    Multiply two matrices (same dimension requirements)
    """
    rowsA = len(matA)
    colsA = len(matA[0])
    rowsB = len(matB)
    colsB = len(matB[0])
    if colsA != rowsB:
        raise ValueError("Matrix dimensions are incompatible for multiplication.")
    multiplied = [[0 for _ in range(colsB)] for _ in range(rowsA)]
    for r in range(rowsA):
        for c in range(colsB):
            for k in range(rowsB):
                multiplied[r][c] += matA[r][k] * matB[k][c]
    return multiplied

def buildIdentityMat(cols, rows):
    """
    Create an identity matrix of size rows x cols
    """
    return [[1 if x == y else 0 for y in range(cols)] for x in range(rows)]

def matVectorMult(invMat, bVec):
    """
    Multiply inverse matrix by vector
    """
    resVec = []
    for i in range(len(bVec)):
        resVec.append([0])
    for i in range(len(invMat)):
        for k in range(len(bVec)):
            resVec[i][0] += invMat[i][k] * bVec[k][0]
    return resVec

def exchangeRowIfPivotZero(mtx, vec):
    """
    If pivot is zero, swap rows in both matrix and vector
    """
    for i in range(len(mtx)):
        for j in range(i, len(mtx)):
            if mtx[i][i] == 0:
                tempRow = mtx[j]
                tempValB = vec[j]
                mtx[j] = mtx[i]
                vec[j] = vec[i]
                mtx[i] = tempRow
                vec[i] = tempValB
    return [mtx, vec]

def calcCondition(mtx, invMtx):
    """
    Calculate condition number: ||A|| * ||A(-1)||
    """
    print("|| A ||max =", maxRowSumNorm(mtx))
    print("|| A(-1) ||max =", maxRowSumNorm(invMtx))
    return maxRowSumNorm(mtx) * maxRowSumNorm(invMtx)

def invertMatrix(mtx, vec=None):
    """
    Compute matrix inverse using Gauss-Jordan approach
    """
    if calcDeterminant(mtx, 1) == 0:
        print("Error, Singular Matrix\n")
        return
    invRes = buildIdentityMat(len(mtx), len(mtx))
    for pivotIdx in range(len(mtx[0])):
        # החלפת שורות אם צריך
        mtx, vec = swapPivotRow(mtx, vec)
        eMat = buildIdentityMat(len(mtx[0]), len(mtx))
        eMat[pivotIdx][pivotIdx] = 1 / mtx[pivotIdx][pivotIdx]
        invRes = matMult(eMat, invRes)
        mtx = matMult(eMat, mtx)
        for downRow in range(pivotIdx + 1, len(mtx)):
            eMat = buildIdentityMat(len(mtx[0]), len(mtx))
            eMat[downRow][pivotIdx] = -(mtx[downRow][pivotIdx])
            mtx = matMult(eMat, mtx)
            invRes = matMult(eMat, invRes)

    for pivotIdx in range(len(mtx[0]) - 1, 0, -1):
        for upRow in range(pivotIdx - 1, -1, -1):
            eMat = buildIdentityMat(len(mtx[0]), len(mtx))
            eMat[upRow][pivotIdx] = -(mtx[upRow][pivotIdx])
            mtx = matMult(eMat, mtx)
            invRes = matMult(eMat, invRes)

    return invRes

def swapPivotRow(mtx, vec):
    """
    Pivoting process that ensures the pivot is the largest element in the column
    """
    for i in range(len(mtx)):
        largestPivot = abs(mtx[i][i])
        for j in range(i, len(mtx)):
            if abs(mtx[j][i]) > largestPivot:
                tempRow = mtx[j]
                tempValB = vec[j]
                mtx[j] = mtx[i]
                vec[j] = vec[i]
                mtx[i] = tempRow
                vec[i] = tempValB
                largestPivot = abs(mtx[i][i])
    return [mtx, vec]
