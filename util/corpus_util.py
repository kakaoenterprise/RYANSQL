import numpy as np
import random

import sys

def shuffleData( vec_data ):
    for _ in range( len( vec_data ) ):
        randidx1    = random.randrange( 0, len( vec_data ) )
        randidx2    = random.randrange( 0, len( vec_data ) )
        tmpdata     = vec_data[ randidx1 ]
        vec_data[ randidx1 ] = vec_data[ randidx2 ]
        vec_data[ randidx2 ] = tmpdata
    
    return vec_data



