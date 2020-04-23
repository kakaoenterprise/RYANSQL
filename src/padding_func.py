import numpy as np

# IN: BS X L1 X L2 X L3 X L4
# OUT: ( BS*L1*L2*L3 ) X L3. Padded / Mask for Dim 1 / Mask for Dim 2 / Mask for Dim 3 / Mask for Dim 4.
def padding_5D_to_2D( vec_target, padding_idx ):
    l1_mask = [ len( dim1 ) for dim1 in vec_target ]
    l1_max  = max( l1_mask )

    l2_mask = np.zeros( len( vec_target ) * l1_max, dtype = int )
    for l1_idx, l1_v in enumerate( vec_target ):
        base_idx    = l1_idx * l1_max
        l2_mask[ base_idx: base_idx + len( l1_v ) ] = [ len( l2_v ) for l2_v in l1_v ] 
    l2_max  = np.max( l2_mask )

    l3_mask = np.zeros( len( vec_target ) * l1_max * l2_max, dtype = int )
    for l1_idx, l1_v in enumerate( vec_target ):
        for l2_idx, l2_v in enumerate( l1_v ):
            base_idx    = l1_idx * l1_max * l2_max + l2_idx * l2_max
            l3_mask[ base_idx : base_idx + len( l2_v ) ]    = [ len( l3_v ) for l3_v in l2_v ]
    l3_max  = np.max( l3_mask )
    
    l4_mask = np.zeros( len( vec_target ) * l1_max * l2_max * l3_max, dtype = int )
    for l1_idx, l1_v in enumerate( vec_target ):
        for l2_idx, l2_v in enumerate( l1_v ):
            for l3_idx, l3_v in enumerate( l2_v ):
                base_idx    = l1_idx * l1_max * l2_max * l3_max + l2_idx * l2_max * l3_max + l3_idx * l3_max
                l4_mask[ base_idx : base_idx + len( l3_v ) ]    = [ len( l4_v ) for l4_v in l3_v ]
    l4_max  = np.max( l4_mask )

    # Generate a huge np array with masks.
    padded_ret  = np.ones( ( len( vec_target ) * l1_max * l2_max * l3_max, l4_max ) ) * padding_idx
    for l1_idx, l1_v in enumerate( vec_target ):
        for l2_idx, l2_v in enumerate( l1_v ):
            for l3_idx, l3_v in enumerate( l2_v ):
                base_idx    = l1_idx * l1_max * l2_max * l3_max + l2_idx * l2_max * l3_max + l3_idx * l3_max
                for l4_idx, l4_v in enumerate( l3_v ):
                    padded_ret[ base_idx + l4_idx, 0: len( l4_v ) ] = l4_v

    return padded_ret, l1_mask, l2_mask, l3_mask, l4_mask

# IN: BS X L1 X L2 X L3
# OUT: ( BS*L1*L2 ) X L3. Padded / Mask for Dim 1 / Mask for Dim 2 / Mask for Dim 3.
def padding_4D_to_2D( vec_target, padding_idx ):
    l1_mask = [ len( dim1 ) for dim1 in vec_target ]
    l1_max  = max( l1_mask )

    l2_mask = np.zeros( len( vec_target ) * l1_max, dtype = int )
    for l1_idx, l1_v in enumerate( vec_target ):
        base_idx    = l1_idx * l1_max
        l2_mask[ base_idx: base_idx + len( l1_v ) ] = [ len( l2_v ) for l2_v in l1_v ] 
    l2_max  = np.max( l2_mask )

    l3_mask = np.zeros( len( vec_target ) * l1_max * l2_max, dtype = int )
    for l1_idx, l1_v in enumerate( vec_target ):
        for l2_idx, l2_v in enumerate( l1_v ):
            base_idx    = l1_idx * l1_max * l2_max + l2_idx * l2_max
            l3_mask[ base_idx : base_idx + len( l2_v ) ]    = [ len( l3_v ) for l3_v in l2_v ]
    l3_max  = np.max( l3_mask )

    # Generate a huge np array with masks.
    padded_ret  = np.ones( ( len( vec_target ) * l1_max * l2_max, l3_max ) ) * padding_idx
    for l1_idx, l1_v in enumerate( vec_target ):
        for l2_idx, l2_v in enumerate( l1_v ):
            base_idx    = l1_idx * l1_max * l2_max + l2_idx * l2_max
            for l3_idx, l3_v in enumerate( l2_v ):
                padded_ret[ base_idx + l3_idx, 0: len( l3_v ) ] = l3_v

    return padded_ret, l1_mask, l2_mask, l3_mask

# IN: 
#   vec_target: BS X L1 X L2.
# OUT: ( BS*L1 ) X L2. Padded / The length mask vector.
def padding_3D_to_2D( vec_target, padding_idx ):
    l1_mask = [ len( dim1 ) for dim1 in vec_target ]
    l1_max  = max( l1_mask )

    l2_mask = np.zeros( len( vec_target ) * l1_max, dtype = int )
    for l1_idx, l1_v in enumerate( vec_target ):
        base_idx    = l1_idx * l1_max
        l2_mask[ base_idx: base_idx + len( l1_v ) ] = [ len( l2_v ) for l2_v in l1_v ] 
    l2_max  = np.max( l2_mask )

    # Generate a huge np array with masks.
    padded_ret  = np.ones( ( len( vec_target ) * l1_max, l2_max ) ) * padding_idx
    for l1_idx, l1_v in enumerate( vec_target ):
        base_idx    = l1_idx * l1_max 
        for l2_idx, l2_v in enumerate( l1_v ):
            padded_ret[ base_idx + l2_idx, 0: len( l2_v ) ] = l2_v

    return padded_ret, l1_mask, l2_mask

# IN: BS X T.
# max_len: T.
# OUT: BS X T. ( PADDED )
def padding_2D( vec_target, padding_idx ):
    l1_mask = [ len( dim1 ) for dim1 in vec_target ]
    l1_max  = max( l1_mask )

    # Generate a huge np array with masks.
    padded_ret  = np.ones( ( len( vec_target ),  l1_max ) ) * padding_idx
    for l1_idx, l1_v in enumerate( vec_target ):
        padded_ret[ l1_idx, 0: len( l1_v ) ] = l1_v

    return padded_ret, l1_mask

# IN: BS X T.
# OUT: BS X T. ( PADDED )
def padding_2D_len( vec_target, max_len, padding_idx ):
    padded_ret  = np.ones( ( len( vec_target ), max_len ) ) * padding_idx
    for batch_idx, batch_v in enumerate( vec_target ):
        padded_ret[ batch_idx, 0: len( batch_v ) ]  = batch_v
    return padded_ret

def make_one_hot( vec_target, max_len ):
    ret = np.zeros( max_len, dtype = float )
    for v in vec_target:
        ret[v]  = 1.0

    return ret

