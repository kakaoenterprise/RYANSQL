import tensorflow as tf
import numpy as np
import math
import random

variable_initializer = tf.contrib.layers.variance_scaling_initializer( factor=2.0, mode='FAN_IN', uniform = False, dtype = tf.float32 )

def msra_initializer( factor_scale = 1.0 ):
    return tf.contrib.layers.variance_scaling_initializer( factor = 2.0 * factor_scale, mode = "FAN_IN", uniform = False, dtype = tf.float32 )


# inputs = BS X D1 X D2, begin_norm_axis is assumed to be 1.
# specialized layer_norm for masked case.
# Assumed Masked value, Returns masked value.
def layer_norm( inputs, mask, variance_epsilon = 1e-12, regularizer = None, reuse = None, scope = None ):
    with tf.variable_scope( scope, 'LayerNorm', [ inputs ], reuse = reuse ) as sc:
        inputs_shape    = inputs.shape
        inputs_rank     = inputs_shape.ndims
        if inputs_rank is None:
            raise ValueError('Inputs %s has undefined rank.' % inputs.name)
        mask_rank       = mask.shape.ndims 

        params_shape= inputs_shape[ -1: ] 
        beta        = tf.get_variable( "beta", shape = params_shape, dtype = tf.float32, initializer = tf.zeros_initializer(), trainable = True, regularizer = regularizer )
        gamma       = tf.get_variable( "gamma", shape = params_shape, dtype = tf.float32, initializer = tf.ones_initializer(), trainable = True, regularizer = regularizer )

        # Calculate means & Variances.
        # Assume the input is already masked.
        plain_num   = tf.cast( tf.shape( inputs )[1] * tf.shape( inputs )[2], tf.float32 )
        if mask_rank == 1:
            actual_num  = tf.cast( mask * tf.shape( inputs )[2], tf.float32 )
        else:
            actual_num  = tf.cast( tf.reduce_sum( tf.squeeze( tf.cast( mask, tf.int32 ), -1 ), 1 ) * tf.shape( inputs )[2], tf.float32 ) # BS. 0 value is possible.

        ratio       = actual_num / plain_num                                            # BS
        ratio       = tf.expand_dims( tf.expand_dims( ratio, 1 ), 1 )
        mean        = tf.reduce_mean( inputs, axis = [ 1, 2 ], keep_dims = True, name = "mean" )    # BS X 1 X 1
        mean        = tf.where( tf.cast( tf.expand_dims( tf.expand_dims( actual_num, -1 ), -1 ), tf.bool ), mean / ratio, tf.zeros_like( mean ) )
#        mean        /= ratio

        sq_diff     = tf.squared_difference( inputs, tf.stop_gradient( mean ) )
        # Zero-out using masks.
        if mask_rank == 1:
            mask_key    = tf.sequence_mask( mask, tf.shape( inputs )[1] )   # BS X D1
            mask_key    = tf.expand_dims( mask_key, 2 ) # BS X D1 X 1
        else:
            mask_key    = mask

        mask_key    = tf.tile( mask_key, [ 1, 1, tf.shape( inputs )[2] ] ) # BS X D1 X D2
        mask_val    = tf.zeros_like( sq_diff )
        sq_diff     = tf.where( mask_key, sq_diff, mask_val )

        var = tf.reduce_mean( sq_diff, [ 1, 2 ], keep_dims = True, name = "var" )   # BS X 1 X 1
        var = tf.where( tf.cast( tf.expand_dims( tf.expand_dims( actual_num, -1 ), -1 ), tf.bool ), var / ratio, tf.zeros_like( mean ) )
#        var /= ratio

        normalized_input    = ( inputs - mean ) / tf.sqrt( var + variance_epsilon )     
        normalized_input    = tf.multiply( normalized_input, gamma ) + beta
        normalized_input    = tf.where( mask_key, normalized_input, mask_val )
        return normalized_input

def layer_dropout( inputs, residual, dropout_rate ):
    pred    = tf.random_uniform( [] ) < dropout_rate
    return tf.cond( pred, lambda: residual, lambda: inputs + residual )

def layer_dropout_both( inputs, residual, dropout_rate ):
    rand    = tf.random_uniform( [] )
    return tf.cond( rand < dropout_rate, \
                    lambda: inputs, \
                    lambda: tf.cond( rand < 2 * dropout_rate, lambda: residual, lambda: inputs + residual ) )

def conv( inputs, output_size, bias = None, activation = None, kernel_size = 1, name = "conv", reuse = None, regularizer = None, padding = "VALID" ):
    with tf.variable_scope(name, reuse = reuse):
        shapes = inputs.shape.as_list()
        if len(shapes) > 4:
            raise NotImplementedError
        elif len(shapes) == 4:
            filter_shape = [1,kernel_size,shapes[-1],output_size]
            bias_shape = [1,1,1,output_size]
            strides = [1,1,1,1]
        else:
            filter_shape = [kernel_size,shapes[-1],output_size]
            bias_shape = [1,1,output_size]
            strides = 1
        conv_func = tf.nn.conv1d if len(shapes) == 3 else tf.nn.conv2d
        kernel_ = tf.get_variable("kernel_",
                        filter_shape,
                        dtype = tf.float32,
                        regularizer=regularizer,
                        initializer = variable_initializer )
        outputs = conv_func( inputs, kernel_, strides, padding )
        if bias:
            outputs += tf.get_variable("bias_",
                        bias_shape,
                        regularizer=regularizer,
                        initializer = tf.zeros_initializer())
        if activation is not None:
            return activation(outputs)
        else:
            return outputs

def highway(x, mask, size = None, activation = None, regularizer = None, num_layers = 2, scope = "highway", keep_prob = 1.0, reuse = None ):
    with tf.variable_scope(scope, reuse):
        if size is None:
            size = x.shape.as_list()[-1]
        else:
#            x   = layer_norm( x, mask, regularizer = regularizer, scope = "highway_ln", reuse = reuse ) # Remove this in "FIX MORE" version
            x   = conv(x, size, name = "input_projection", reuse = reuse, regularizer = regularizer)
        
        mask_key    = tf.sequence_mask( mask, tf.shape( x )[1] )   # BS X Q 
        mask_key    = tf.tile( tf.expand_dims( mask_key, 2 ), [ 1, 1, size ] )
        mask_val    = tf.zeros_like( mask_key )
        mask_val    = tf.cast( mask_val, tf.float32 )
        x           = tf.where( mask_key, x, mask_val )

        for i in range(num_layers):
            T = conv( x, size, bias = True, activation = tf.sigmoid, name = "gate_%d"%i, reuse = reuse, regularizer = regularizer)
            H = conv( x, size, bias = True, activation = activation, name = "activation_%d"%i, reuse = reuse, regularizer = regularizer)
            H = tf.nn.dropout( H, keep_prob )

            T   = tf.where( mask_key, T, mask_val )
            H   = tf.where( mask_key, H, mask_val )

            x = H * T + x * (1.0 - T)
        return x

def depthwise_separable_convolution(inputs, kernel_size, num_filters,
                                    scope = "depthwise_separable_convolution",
                                    bias = True, is_training = True, reuse = None, regularizer = None, padding = "SAME", activation = tf.nn.relu ):
    with tf.variable_scope(scope, reuse = reuse):
        shapes = inputs.shape.as_list()
        depthwise_filter = tf.get_variable("depthwise_filter",
                                        (kernel_size[0], kernel_size[1], shapes[-1], 1),
                                        dtype = tf.float32,
                                        regularizer=regularizer,
                                        initializer = variable_initializer )
        pointwise_filter = tf.get_variable("pointwise_filter",
                                        (1,1,shapes[-1],num_filters),
                                        dtype = tf.float32,
                                        regularizer=regularizer,
                                        initializer = variable_initializer )
        outputs = tf.nn.separable_conv2d(inputs,
                                        depthwise_filter,
                                        pointwise_filter,
                                        strides = (1,1,1,1),
                                        padding = padding )
        if bias:
            b = tf.get_variable("bias",
                    outputs.shape[-1],
                    regularizer=regularizer,
                    initializer = tf.zeros_initializer())
            outputs += b

        outputs = activation( outputs )
        return outputs


# Apply mask.
# target: BS X L X D
# mask: BS. Mask on L.
def apply_mask( target, mask, mask_seed ):
    if mask.shape.ndims == 1:
        mask_key    = tf.expand_dims( tf.sequence_mask( mask, tf.shape( target )[1] ), -1 )   # BS X L X 1
    else:
        mask_key    = mask
    mask_key    = tf.tile( mask_key, [ 1, 1, tf.shape( target )[-1] ] )
    return tf.where( mask_key, target, mask_seed * tf.ones_like( target ) )

def apply_mask_NCW( target, mask, mask_seed ):
    mask_key    = tf.sequence_mask( mask, tf.shape( target )[2] )   # BS X L
    mask_key    = tf.tile( tf.expand_dims( mask_key, 1 ), [ 1, target.get_shape().as_list()[1], 1 ] )    # BS X D X L
    return tf.where( mask_key, target, mask_seed * tf.ones_like( target ) )

def dense_net_block( mat_in, mask, kernel_size, filter_size, keep_prob, scope, regularizer = None ):
    with tf.variable_scope( scope ):
        conv1d_filter   = tf.get_variable( name = "c1d_filter", initializer = variable_initializer, shape = [ kernel_size, mat_in.get_shape().as_list()[1], filter_size ], regularizer = regularizer, dtype = tf.float32 )
        ret = tf.nn.conv1d( mat_in, conv1d_filter, 1, "SAME", data_format = "NCW" )
        ret = tf.nn.dropout( ret, keep_prob )
        ret = tf.nn.leaky_relu( ret )
        ret = apply_mask_NCW( ret, mask, float( 0.0 ) )
        return ret

# mat_in: BS X D X T
# out: BS X D' X T
def dense_net( conv_num, mat_in, mask, first_dim, other_dim, kernel_size, keep_prob, scope, regularizer = None ):
    vec_results = []
    with tf.variable_scope( scope ):
        # First convolution.
        conv_first  = dense_net_block( mat_in, mask, 1, first_dim, keep_prob, "BLOCK_1", regularizer )  
        vec_results.append( conv_first )

        # Other convolutions.
        for conv_idx in range( 2, conv_num + 1 ):
            conv_result = dense_net_block( tf.concat( vec_results, 1 ), mask, kernel_size, other_dim, keep_prob, "BLOCK_%d" % conv_idx, regularizer )
            vec_results.append( conv_result )

        # Return the concatenations of all the results.
        return tf.concat( vec_results, 1 )

# Reimplementation of DSA: Dynamic Self-Attention: Computing Attention over Words Dynamically for Sentence Embedding
# mat_in: BS X T X D
# Out: BS X D'
def dsa( mat_in, mask, batch_size, t_maxlen, att_num, att_dim, r_iter, scope, regularizer = None ):
    with tf.variable_scope( scope ):
        dsa_target  = tf.layers.dense( mat_in, att_num * att_dim, activation = tf.nn.leaky_relu, kernel_initializer = variable_initializer, name = "DSA", kernel_regularizer = regularizer )   # BS X T X (N*D). [A1, A2, ..., AN ]
        dsa_target  = tf.reshape( dsa_target, [ batch_size, t_maxlen, att_num, att_dim ] )
        dsa_target  = tf.transpose( dsa_target, [ 0, 2, 1, 3 ] )    # BS X ATT_NUM X T X Da

        q_matrix    = tf.zeros_like( mat_in )
        q_matrix    = tf.reduce_mean( q_matrix, [-1], keep_dims = True )
        q_matrix    = tf.tile( q_matrix, [ 1, 1, att_num ] )    # BS X T X ATT_NUM. zero matrix.

        cur_ret     = None
        for r_idx in range( r_iter ):
            att_matrix  = apply_mask( q_matrix, mask, float( "-inf" ) )
            att_matrix  = tf.transpose( att_matrix, [ 0, 2, 1 ] )
            att_matrix  = tf.nn.softmax( att_matrix, -1 )       # BS X ATT_NUM X T.
            att_matrix  = tf.expand_dims( att_matrix, 2 )       # BS X ATT_NUM X 1 X T.

            s           = tf.matmul( att_matrix, dsa_target )   # BS x ATT_NUM X 1 X Da.
            z           = tf.tanh( s )
            cur_ret     = tf.reshape( z, [ batch_size, att_num * att_dim ] )

            q_update    = tf.matmul( dsa_target, z, transpose_b = True )    # BS X ATT_NUM X T X 1.
            q_update    = tf.transpose( tf.squeeze( q_update, -1 ), [ 0, 2, 1 ] )   # BS x T X ATT_NUM
            q_matrix    = q_matrix + q_update

        return cur_ret



