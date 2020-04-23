import sys
sys.path.append( "../../util" )

from networkcomponents  import *
from db_meta            import *
from multiple_pointer_custom   import *

# INPUT:
# start_context: BS X N X D1
# t_col_enoded: BS X T X D2.
# Output:
# 1. Columns of the VU units. BS X N X T
# 2. Context Mask: Actual # of used contexts. BS
# 2. Operator raw scores of the VU unit. BS X N X Len( OP )
# 3. Distinct raw scores of the VU unit. BS X N X 2
def generate_multiple_vu( start_context, context_mask, t_col_encoded, t_col_mask, iter_num, scope, regularizer, do_agg_tot = False ):
    with tf.variable_scope( scope, reuse = tf.AUTO_REUSE ):
        start_context   = apply_mask( start_context, context_mask, float( 0.0 ) )
        start_context   = layer_norm( start_context, context_mask, scope = "c_ln" )
        mc1, p1         = get_pointer_N( start_context, t_col_encoded, t_col_mask, regularizer, scope = "PTR1", dim_scale = 2 )
        mc1             = apply_mask( mc1, context_mask, float( 0.0 ) )
        mc1             = layer_norm( mc1, context_mask, scope = "m1_ln" )

        # Get Other Scores.
        agg1    = tf.layers.dense( mc1, len( VEC_AGGREGATORS ), kernel_initializer = variable_initializer, name = "agg_proj", kernel_regularizer = regularizer )
        dist1   = tf.layers.dense( mc1, 2, kernel_initializer = variable_initializer, name = "dist_proj", kernel_regularizer = regularizer )
        op      = tf.layers.dense( mc1, len( VEC_OPERATORS ), kernel_initializer = variable_initializer, name = "oper_proj", kernel_regularizer = regularizer )
        aggt    = tf.layers.dense( mc1, len( VEC_AGGREGATORS ), kernel_initializer = variable_initializer, name = "agg_tot_proj", kernel_regularizer = regularizer )

        # Get Col Ptr 2.
        mc2, p2 = get_pointer_N( mc1, t_col_encoded, t_col_mask, regularizer, scope = "PTR2", dim_scale = 2 )
        mc2     = apply_mask( mc2, context_mask, float( 0.0 ) )
        mc2     = layer_norm( mc2, context_mask, scope = "m2_ln" )

        # Get Other scores.
        agg2    = tf.layers.dense( mc2, len( VEC_AGGREGATORS ), kernel_initializer = variable_initializer, name = "agg_proj", kernel_regularizer = regularizer )
        dist2   = tf.layers.dense( mc2, 2, kernel_initializer = variable_initializer, name = "dist_proj", kernel_regularizer = regularizer )
    
    if do_agg_tot:
        return mc1, mc2, p1, p2, agg1, agg2, dist1, dist2, op, aggt

    return mc1, mc2, p1, p2, agg1, agg2, dist1, dist2, op 
    
