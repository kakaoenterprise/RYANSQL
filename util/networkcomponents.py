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
        if len( shapes ) == 4:
            filter_shape = [1,kernel_size,shapes[-1],output_size]
            bias_shape = [1,1,1,output_size]
            strides = [1,1,1,1]
        else:
            filter_shape = [kernel_size,shapes[-1],output_size]
            bias_shape = [1,1,output_size]
            strides = 1

        conv_func   = tf.nn.conv1d if len(shapes) == 3 else tf.nn.conv2d
        k           = tf.get_variable( "k", filter_shape, dtype = tf.float32, regularizer=regularizer, initializer = variable_initializer )
        outputs     = conv_func( inputs, kernel_, strides, padding )

        if bias:
            outputs += tf.get_variable("b", bias_shape, regularizer=regularizer, initializer = tf.zeros_initializer() )
        
        if activation is not None:
            return activation(outputs)
        
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

