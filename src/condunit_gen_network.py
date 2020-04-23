import sys

from networkcomponents  import *
from db_meta            import *
from multiple_pointer_custom   import *
from valueunit_gen_network  import *

def check_nan( t, msg, vals = None, s = 3, rev = False ):
    if vals == None:
        vals    = [ t ]
    v = tf.reduce_sum( t )
      
    if rev:
        t   = tf.cond( tf.math.logical_or( tf.is_nan( v ), tf.is_inf( v ) ), lambda: t, lambda: tf.Print( t, vals, message = msg, summarize = s ) )
    else:
        t   = tf.cond( tf.math.logical_or( tf.is_nan( v ), tf.is_inf( v ) ), lambda: tf.Print( t, vals, message = msg, summarize = s ), lambda: t )
    return t

# INPUT:
# start_context: BS X N X D1
# t_col_enoded: BS X T X D2.
# Output:
# 1. Columns of the VU units. BS X N X T
# 2. Operator raw scores of the VU unit. BS X N X Len( OP )
# 3. Distinct raw scores of the VU unit. BS X N X 2
def generate_multiple_cu( start_context, context_mask, q_col_encoded, t_col_encoded, q_col_mask, t_col_mask, iter_num, scope, regularizer ):
    with tf.variable_scope( scope, reuse = tf.AUTO_REUSE ):
        mc1, mc2, vu_c1, vu_c2, vu_agg1, vu_agg2, vu_dist1, vu_dist2, vu_op = generate_multiple_vu( start_context, context_mask, t_col_encoded, t_col_mask, iter_num, "VU", regularizer )

        mc1 = check_nan( mc1, "mc1" )
        mc2 = check_nan( mc2, "mc2" )


        agg     = tf.layers.dense( mc1, 2, kernel_initializer = variable_initializer, name = "CU_AGG", kernel_regularizer = regularizer )
        is_not  = tf.layers.dense( mc1, 2, kernel_initializer = variable_initializer, name = "CU_IS_NOT", kernel_regularizer = regularizer )
        cond_op = tf.layers.dense( mc1, len( VEC_CONDOPS ), kernel_initializer = variable_initializer, name = "CU_COND_OP", kernel_regularizer = regularizer )
        
        v1_type = tf.layers.dense( mc1, 3, kernel_initializer = variable_initializer, name = "CU_VAL1_TYPE", kernel_regularizer = regularizer )
        v1_like = tf.layers.dense( mc1, 4, kernel_initializer = variable_initializer, name = "CU_VAL1_LIKELY", kernel_regularizer = regularizer )
        v1_bv   = tf.layers.dense( mc1, 2, kernel_initializer = variable_initializer, name = "CU_VAL1_BOOLVAL", kernel_regularizer = regularizer )
        sp1_c, v1_sp    = get_pointer_N( mc1, q_col_encoded, q_col_mask, regularizer, scope = "SP" )
        sp1_c           = apply_mask( sp1_c, context_mask, float( 0.0 ) )
        sp1_c           = layer_norm( sp1_c, context_mask, scope = "ln_sp1" )
        ep1_c, v1_ep    = get_pointer_N( sp1_c, q_col_encoded, q_col_mask, regularizer, scope = "EP" )
        ep1_c           = apply_mask( ep1_c, context_mask, float( 0.0 ) )
        ep1_c           = layer_norm( ep1_c, context_mask, scope = "ln_ep1" )

        sp1_c   = check_nan( sp1_c, "sp1_c" )
        ep1_c   = check_nan( sp1_c, "ep1_c" )


        # V2: Only Pointers are necessary! - NEED TO BE FIXED 
        v2_type = tf.layers.dense( sp1_c, 3, kernel_initializer = variable_initializer, name = "CU_VAL2_TYPE", kernel_regularizer = regularizer )
        v2_like = tf.layers.dense( sp1_c, 4, kernel_initializer = variable_initializer, name = "CU_VAL2_LIKELY", kernel_regularizer = regularizer )
        v2_bv   = tf.layers.dense( sp1_c, 2, kernel_initializer = variable_initializer, name = "CU_VAL2_BOOLVAL", kernel_regularizer = regularizer )
        sp2_c, v2_sp    = get_pointer_N( sp1_c, q_col_encoded, q_col_mask, regularizer, scope = "SP" )
        sp2_c           = apply_mask( sp2_c, context_mask, float( 0.0 ) )
        sp2_c           = layer_norm( sp2_c, context_mask, scope = "ln_sp2" )
        ep2_c, v2_ep    = get_pointer_N( sp2_c, q_col_encoded, q_col_mask, regularizer, scope = "EP" )
        ep2_c           = apply_mask( ep2_c, context_mask, float( 0.0 ) )
        ep2_c           = layer_norm( ep2_c, context_mask, scope = "ln_ep2" )

        return agg, is_not, cond_op, \
                v1_type, v1_like, v1_bv, v1_sp, v1_ep, \
                v2_type, v2_like, v2_bv, v2_sp, v2_ep, \
                vu_c1, vu_c2, vu_agg1, vu_agg2, vu_dist1, vu_dist2, vu_op
