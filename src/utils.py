import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

#define I 
def I(l):
    return np.identity(l)


def find_nonzero_row(matrix, pivot_row, col):
    "While doing GE on matrix - find first nonzero row"
    nrows = matrix.shape[0] 
    for row in range(pivot_row, nrows): 
        if matrix[row, col] != 0: 
            return row 
    return None


def row_echelon_form(mat,cmat=None,rref=True): 
    """
    Get REF form of matrix
    Args:
        mat: the matrix to be reduced
    
    Returns:
        REF matrix and transformation matrix
    """
    matrix = mat.copy()

    pivot_row = 0
    if cmat is None: cmat = I(matrix.shape[0])

    for col in range(matrix.shape[1]): 
        nonzero_row = find_nonzero_row(matrix, pivot_row, col) 
        
        if nonzero_row is not None: 
            matrix[[pivot_row, nonzero_row]] = matrix[[nonzero_row, pivot_row]] #swap rows
            cmat[[pivot_row, nonzero_row]] = cmat[[nonzero_row, pivot_row]] 

            for row in range(pivot_row + 1, matrix.shape[0]): # eliminate below rows
                factor = matrix[row, col] 
                matrix[row] -= (factor * matrix[pivot_row] ) % 2
                cmat[row] -= ( factor *  cmat[pivot_row] ) % 2

            matrix = matrix % 2
            cmat = cmat % 2
            pivot_row += 1
    
    if rref: #converting matrix into reduced row echelon form
        for row in range(matrix.shape[0]):
            for col in range(matrix.shape[1]): 
                if matrix[row,col] == 1:
                    for cand_row in range(0,row):
                        if matrix[cand_row,col] == 1:
                            matrix[cand_row] -= matrix[row]
                            cmat[cand_row] -= cmat[row]
                            
                            matrix = matrix % 2
                            cmat = cmat % 2
                    break

    return matrix,cmat


#visulaization functions
def showMatrix(m, fheight=10,annote=False,block=False, **kwargs):
    aspDiv = 1
    if len(m.shape) > 1 and m.shape[0] < m.shape[1]/2: aspDiv = 2
    plt.figure(figsize=(fheight,fheight/aspDiv))
    ax = sns.heatmap(m, linewidth=0.1, square=True,cmap='BuPu',annot=annote) 
    if 'title' in kwargs:
        plt.title(kwargs.get('title'))
    plt.show(block=block)


def cross_prod(pt1:tuple, pt2:tuple, pt3: tuple) -> float:
    val = ((pt2[0] - pt1[0])*(pt3[1] - pt1[1]) - (pt2[1] - pt1[1])*(pt3[0] - pt1[0]))

    return val.astype(np.float16)