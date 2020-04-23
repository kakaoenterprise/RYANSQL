from networkcomponents  import *

# Get a pointer based on the previous context.
# Update the context using the pointer retrieved.
# prev_context: BS X N X D
# pointer_target: BS X T X D
# Returns
# merged_context: BS X N X D.
# pointer: BS X N X T
def get_pointer_N( prev_context, pointer_target, pointer_mask, regularizer, scope, dim_scale = 1 ):
    with tf.variable_scope( scope, reuse = tf.AUTO_REUSE ):
        dim             = pointer_target.get_shape().as_list()[-1]
        pointer         = tf.tanh( tf.layers.dense( tf.expand_dims( pointer_target, 1 ), dim * dim_scale, kernel_initializer = variable_initializer, name = "att1", use_bias = False, kernel_regularizer = regularizer ) + \
                                    tf.layers.dense( tf.expand_dims( prev_context, 2 ), dim * dim_scale, kernel_initializer = variable_initializer, name = "att2", use_bias = False, kernel_regularizer = regularizer ) )
        pointer         = tf.layers.dense( pointer, 1, kernel_initializer = variable_initializer, name = "attF", use_bias = False, kernel_regularizer = regularizer )   # BS X N X T X 1.
        pointer         = tf.transpose( tf.squeeze( pointer, -1 ), [ 0, 2, 1 ] )    # BS X T X N.
        pointer         = apply_mask( pointer, pointer_mask, float( "-inf" ) )

        # Update context.
        score           = tf.nn.softmax( tf.transpose( pointer, [ 0, 2, 1 ] ) ) # BS X N X T.
        updated_p       = tf.matmul( score, pointer_target )    # BS X N X D.
    
        merged_context  = tf.concat( [ prev_context, updated_p, tf.abs( prev_context - updated_p ), tf.multiply( prev_context, updated_p ) ], -1 )
        merged_context  = tf.layers.dense( merged_context, dim, kernel_initializer = variable_initializer, name = "ctx_proj", use_bias = False, kernel_regularizer = regularizer )

        return merged_context, tf.transpose( pointer, [ 0, 2, 1 ] )

