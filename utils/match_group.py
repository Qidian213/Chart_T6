import numpy as np
from munkres import Munkres,print_matrix

def py_max_match(scores):
    m = Munkres()
    tmp = m.compute(scores)
    tmp = np.array(tmp).astype(np.int32)
    return tmp
    
# matrix = [[5, 9],
          # [10, 7],
          # [8, 1]
        # ]

# indexes = py_max_match(matrix)
# print_matrix(matrix, msg='Lowest cost through this matrix:')

# total = 0
# for row, column in indexes:
    # value = matrix[row][column]
    # total += value
    # print(f'({row}, {column}) -> {value}')
# print(f'total cost: {total}')
