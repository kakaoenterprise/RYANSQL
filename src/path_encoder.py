import tensorflow as tf
import sys
sys.path.append( "../../util" )

from networkcomponents  import *
from db_meta            import *

class PathEncoder:
    def __init__( self ):
        vec_pathcode        = [ PATH_NONE, PATH_UNION, PATH_INTER, PATH_EXCEPT, PATH_WHERE, PATH_HAVING, PATH_PAR ]
        self.path_dic       = dict()
        self.path_dic_rev   = dict()
        for pidx, p in enumerate( vec_pathcode ):
            self.path_dic[ p ]          = pidx
            self.path_dic_rev[ pidx ]   = p

    def _initialize_embeddings( self, path_embed_len, regularizer = None ):
        print ( "PATH INITEMBED CALLED" )
        path_embed      = tf.get_variable( name = "path_embed", initializer = variable_initializer, shape = [ len( self.path_dic ), path_embed_len ], trainable = True, regularizer = regularizer, dtype = tf.float32 )
        path_pad        = tf.constant( 0.0, shape = [ 1, path_embed_len ], dtype = tf.float32 )
        self.path_embed = tf.concat( [ path_embed, path_pad ], 0 )

    # path_idx: BS X P.
    # Returns: BS X D. path encoding.
    def get_path_embeddings( self, path_embed_len, \
                            path_idx, path_mask, \
                            final_out_dim, scope, \
                            training, keep_prob = 1.0, regularizer = None ):
        with tf.variable_scope( scope ):
            self._initialize_embeddings( path_embed_len, regularizer )

            batch_size  = tf.shape( path_idx )[0]
            max_p_len   = tf.shape( path_idx )[1]   

            # 1. Convert to path embed matrix.
            p_embed = tf.nn.embedding_lookup( self.path_embed, path_idx )
            p_embed = tf.nn.dropout( p_embed, keep_prob )
            p_embed = apply_mask( p_embed, path_mask, float( 0.0 ) )


            conv1d_filter   = tf.get_variable( name = "c1d_filter", initializer = variable_initializer, shape = [ 3, p_embed.get_shape().as_list()[-1], p_embed.get_shape().as_list()[-1]], regularizer = regularizer, dtype = tf.float32 )

            p_embed = tf.nn.conv1d( p_embed, conv1d_filter, 1, "SAME" )
            p_embed = tf.nn.dropout( p_embed, keep_prob )
            p_embed = tf.nn.leaky_relu( p_embed )
            p_embed = apply_mask( p_embed, path_mask, float( "-inf" ) )
            p_embed = tf.reduce_max( p_embed, 1 )
   
        return p_embed

