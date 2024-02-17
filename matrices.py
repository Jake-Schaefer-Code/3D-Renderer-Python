import numpy as np

def _create_orthographic_matrix(r:float,l:float,t:float,b:float) -> np.ndarray:
    # Scales x and y values to between -1 and 1
    scaleM = np.array([[2/(r-l),  0,      0,    0 ],
                       [ 0,     2/(t-b),  0,    0 ],
                       [ 0,       0,      1,    0 ],
                       [ 0,       0,      0,    1 ]], dtype='float64')
    
    # Translates the viewing frustum apex to the origin
    transM = np.array([[ 1,   0,   0,  -(r+l)/2 ],
                       [ 0,   1,   0,  -(t+b)/2 ],
                       [ 0,   0,   1,      0    ],
                       [ 0,   0,   0,      1    ]], dtype='float64')
    
    orthoM = scaleM @ transM
    return orthoM

def _create_perspective_matrix(n:float, f:float) -> np.ndarray:
    # Scales z such that the near and far plane are -1 and 1 respectively
    perspM = np.array([[ n,   0,   0,   0 ],
                       [ 0,   n,   0,   0 ],
                       [ 0,   0,  -(n+f)/(n-f), 2*n*f/(n-f)], 
                       [ 0,   0,   1,   0 ]], dtype='float64')
    return perspM


def create_projection_matrix(lbn:np.ndarray, rtf:np.ndarray) -> np.ndarray:
    l,b,n = lbn
    r,t,f = rtf
    orthoM = _create_orthographic_matrix(r,l,t,b)
    perspM = _create_perspective_matrix(n,f)
    projM = orthoM @ perspM
    return projM

TRANSLATION_MATRIX = np.identity(4)
def translate(matrix: np.ndarray, trans_vector: np.ndarray) -> np.ndarray:
    TRANSLATION_MATRIX[0][3] = -trans_vector[0]
    TRANSLATION_MATRIX[1][3] = -trans_vector[1]
    TRANSLATION_MATRIX[2][3] = -trans_vector[2]
    return TRANSLATION_MATRIX @ matrix