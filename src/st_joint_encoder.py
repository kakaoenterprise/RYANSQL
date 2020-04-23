import tensorflow as tf
import sys

from networkcomponents  import *

import modeling

class STJointEncoderBert:
    def __init__( self, bert_config ):
        self._bert_config   = bert_config

    def _initialize_embeddings( self, regularizer = None ):
        tp_embed    = tf.get_variable( name = "primary_key_embed", initializer = variable_initializer, shape = [ 2, 50 ], trainable = True, regularizer = regularizer, dtype = tf.float32 )
        tp_pad      = tf.constant( 0.0, shape = [ 1, 50 ], dtype = tf.float32 )
        self.pk_embed   = tf.concat( [ tp_embed, tp_pad ], 0 )
    
        tf_embed    = tf.get_variable( name = "foreign_key_embed", initializer = variable_initializer, shape = [ 2, 50 ], trainable = True, regularizer = regularizer, dtype = tf.float32 )
        tf_pad      = tf.constant( 0.0, shape = [ 1, 50 ], dtype = tf.float32 )
        self.fk_embed   = tf.concat( [ tf_embed, tf_pad ], 0 )
        
    # bert_ids: BS X T. [CLS] qtoks [SEP] col1 [SEP] col2 ... [SEP]
    # bert_mask: BS. length of the bert tokens.
    # bert_seg_id: BS X T. bert segment ID.
    # q_mask: BS. length of the questions.
    # col_loc: BS X C. Index of the "SEP" token for each column. "QUESTION/CLS CONSIDERED"
    # col_mask: BS. Number of Columns.
    def get_bert_embedding( self, bert_ids, bert_mask, bert_seg_id, q_mask, col_loc, col_mask, is_training, scope, keep_prob, regularizer ):
        bert_model  = modeling.BertModel( \
                            scope = scope, \
                            config = self._bert_config, \
                            is_training = is_training, \
                            input_ids   = bert_ids, \
                            input_mask  = tf.cast( tf.sequence_mask( bert_mask, tf.shape( bert_ids )[1] ), tf.int32 ), \
                            token_type_ids = bert_seg_id, \
                            use_one_hot_embeddings = False )

        final_hidden    = bert_model.get_sequence_output()  # BS X T X D.
    
        # 1. Whole DB + Question Encoding.
        db_encoded      = final_hidden[:, 0, :]     # BS X D.

        # 2. Whole Question Encoding + Question Token Encoding.
        q_tok_encoded   = final_hidden[:, 1:, :]
        q_tok_encoded   = apply_mask( q_tok_encoded, q_mask, float( 0.0 ) )  # Including [SEP].
        q_encoded       = self._self_att( q_tok_encoded, q_mask, keep_prob, regularizer, "self_att" )
        q_encoded       = tf.squeeze( q_encoded, 1 )

        # 3. Column Embedding.
        col_loc         = tf.expand_dims( col_loc, -1 ) # BS X C X 1.
        col_encoded     = tf.gather_nd( final_hidden, col_loc, batch_dims = 1 )     # BS X C X D.
        col_encoded     = apply_mask( col_encoded, col_mask, float( 0.0 ) )

        # Returns.
        return q_tok_encoded, col_encoded, q_encoded, db_encoded


    def get_joint_embedding( self, \
        bert_ids, bert_mask, bert_seg_id, \
        q_mask, col_loc, col_mask, \
        is_training, scope, keep_prob = 1.0, regularizer = None ):

        q_tok_encoded, col_encoded, q_encoded, db_encoded   = \
            self.get_bert_embedding( bert_ids, bert_mask, bert_seg_id, q_mask, col_loc, col_mask, is_training, "bert", keep_prob, regularizer )

        return q_tok_encoded, col_encoded, q_encoded, db_encoded
   
    def do_proj( self, mat, mask, proj_dim, regularizer = None ):
        ret = apply_mask( mat, mask, float( 0.0 ) )
        if ret.get_shape().as_list()[-1] == proj_dim:
            return ret

        ret = conv( ret, proj_dim, name = "proj", regularizer = regularizer )
        return ret

    # IN: BS X T X D
    # OUT: BS X 1 X D
    def _self_att( self, v, mask, keep_prob, regularizer, scope ):
        with tf.variable_scope( scope, reuse = tf.AUTO_REUSE ):
            v_att   = tf.tanh( tf.layers.dense( v, v.get_shape().as_list()[-1], kernel_initializer = variable_initializer, name = "att_1", kernel_regularizer = regularizer ) )
            v_att   = tf.tanh( tf.layers.dense( v_att, 1, kernel_initializer = variable_initializer, name = "att_2", kernel_regularizer = regularizer ) )

            v_att   = apply_mask( v_att, mask, float( "-inf" ) )
            v_att   = tf.nn.softmax( tf.transpose( v_att, [ 0, 2, 1 ] ), -1 )     # BS X 1 X C
            ret     = tf.matmul( v_att, v ) 

        return ret

    # tbl_col_idx: Table column index. [ [ [ 1, 2, 3 ], [ 4, 5, 6, 7, 8 ], ... ] ]. BS X T X C. 
    # tbl_col_mask: Table Column Number. [ [ 3, 4 ], [ 2, 3, 6 ],... ]. BS X T.
    # tbl_mask: Number of Tables. BS.
    def get_tbl_joint_embedding( self, \
        bert_ids, bert_mask, bert_seg_id, \
        q_mask, col_loc, col_mask, \
        tbl_col_idx, tbl_col_mask, tbl_mask, \
        is_training, scope, keep_prob = 1.0, regularizer = None ):
    
        q_tok_encoded, col_encoded, q_encoded, db_encoded   = \
            self.get_bert_embedding( bert_ids, bert_mask, bert_seg_id, q_mask, col_loc, col_mask, is_training, "bert", keep_prob, regularizer )

        max_t   = tf.shape( tbl_col_idx )[1] 
        max_c   = tf.shape( tbl_col_idx )[2] 
        dim     = col_encoded.get_shape().as_list()[-1]

        with tf.variable_scope( scope, reuse = tf.AUTO_REUSE ):
            # 1. Extract table info using the column info.
            tbl_col_idx     = tf.expand_dims( tbl_col_idx, -1 )
            tbl_col_encoded = tf.gather_nd( col_encoded, tbl_col_idx, batch_dims = 1 )  # BS X T X C X D.
            tbl_col_encoded = apply_mask( tf.reshape( tbl_col_encoded, [ -1, max_c, dim ] ), tf.reshape( tbl_col_mask, [ -1 ] ), float( 0.0 ) )
            tbl_encoded     = self._self_att( tbl_col_encoded, tf.reshape( tbl_col_mask, [ -1 ] ), keep_prob, regularizer, scope = "column_to_tbl" )
            tbl_encoded     = tf.reshape( tbl_encoded, [ -1, max_t, dim ] )  # BS x T X D.
            tbl_encoded     = apply_mask( tbl_encoded, tbl_mask, float( 0.0 ) )
            tbl_encoded     = layer_norm( tbl_encoded, tbl_mask, scope = "tbl_ln" )
            tbl_encoded     = tf.nn.dropout( tbl_encoded, keep_prob )

        return q_encoded, tbl_encoded
