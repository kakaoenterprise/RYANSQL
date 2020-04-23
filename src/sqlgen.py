import tensorflow as tf
import numpy as np
import pickle
import math
import random
import ast
import sys
sys.path.append( "../../util" )
sys.path.append( "../../util/bert" )

from st_joint_encoder   import STJointEncoderBert
from path_encoder       import PathEncoder
from networkcomponents  import *
from db_meta            import *
from multiple_pointer_custom   import *
from valueunit_gen_network  import *
from condunit_gen_network   import *

import modeling

class SQLGen:
    def __init__( self, bert_config = None ):
        self.je     = STJointEncoderBert( bert_config )
        self.pe     = PathEncoder()
    
    def _check_nan( self, t, msg, vals = None, s = 3, rev = False ):
        if vals == None:
            vals    = [ t ]
        v = tf.reduce_sum( t )
      
        if rev:
            t   = tf.cond( tf.math.logical_or( tf.is_nan( v ), tf.is_inf( v ) ), lambda: t, lambda: tf.Print( t, vals, message = msg, summarize = s ) )
        else:
            t   = tf.cond( tf.math.logical_or( tf.is_nan( v ), tf.is_inf( v ) ), lambda: tf.Print( t, vals, message = msg, summarize = s ), lambda: t )
        return t


    # Get conditioned loss.
    # Gather losses for only those with cond_tensor = 1.
    # cond_tensor = BS, score = BS X C, label = BS
    def _get_conditioned_loss( self, cond_tensor, score, label, loss_func = tf.nn.sparse_softmax_cross_entropy_with_logits ):
        cond_match_num  = tf.cast( tf.reduce_sum( tf.cast( cond_tensor, tf.int32 ) ), tf.float32 )
        loss_sum        = tf.reduce_sum( tf.where( tf.cast( cond_tensor, tf.bool ), loss_func( logits = score, labels = label ), tf.cast( tf.zeros_like( label ), tf.float32 ) ) )
        loss            = tf.cond( cond_match_num > 0, lambda: loss_sum / cond_match_num, lambda: 0.0 )
        
        return loss

    # mat_in: BS X D.
    def _prop_class_net( self, mat_in, class_num, scope, keep_prob, regularizer = None ):
        with tf.variable_scope( scope ):
            score       = tf.layers.dense( mat_in, 256, kernel_initializer = variable_initializer, name = "classification1", kernel_regularizer = regularizer, activation = tf.nn.leaky_relu )
            score       = tf.nn.dropout( score, keep_prob )
            score       = tf.layers.dense( score, class_num, kernel_initializer = variable_initializer, name = "classification2", kernel_regularizer = regularizer )
            result      = tf.argmax( score, -1 )    # BS

            return result, score
   
    def _l2_norm( self, mat, mask ):
        div = tf.reduce_sum( tf.multiply( mat, mat ), -1, keepdims = True )
        div1 = self._check_nan( div, "DIV NAN - 1", vals = [ div ], s = 100 )
        div = tf.sqrt( div1 )
        div = self._check_nan( div, "DIV NAN - 2", vals = [ div, div1 ], s = 100 )

        mat_mask    = tf.sequence_mask( mask, tf.shape( mat )[1] )
        mat_mask    = tf.tile( tf.expand_dims( mat_mask, -1 ), [ 1, 1, mat.get_shape().as_list()[-1] ] )
        
        mat = tf.where( mat_mask, mat / div, tf.zeros_like( mat ) )
        mat = self._check_nan( mat, "MAT NAN", vals = [ mat ], s = 100 )

        return mat

    def constructGraph( self, batch_num = 0, initial_lf = 1e-3, init_bert = None ):
        print ( "CONSTRUCTGRAPH CALLED" )
        self.graph  = tf.Graph()
        with self.graph.as_default():
            with tf.device( "/gpu:0" ):
                ####################
                # BERT INFORMATION #
                ####################
                self.bert_idx       = tf.placeholder( tf.int32, name = BERT_ID, shape = [ None, None ] )    # BS X T.
                self.bert_mask      = tf.placeholder( tf.int32, name = "bert_mask", shape = [ None ] )
                self.bert_seg_id    = tf.placeholder( tf.int32, name = "bert_seg_id", shape = [ None, None ] )
                self.bert_q_mask    = tf.placeholder( tf.int32, name = "bert_q_mask", shape = [ None ] )
                self.bert_col_loc   = tf.placeholder( tf.int32, name = "bert_col_loc", shape = [ None, None ] ) # BS X C.
                self.bert_col_mask  = tf.placeholder( tf.int32, name = "bert_col_mask", shape = [ None ] )

                # BERT for TBL.
                self.bert_tbl_col_idx   = tf.placeholder( tf.int32, name = "bert_tbl_col_idx", shape = [ None, None ] )     # BS*T X C.
                self.bert_tbl_col_mask  = tf.placeholder( tf.int32, name = "bert_tbl_col_mask", shape = [ None, None ] )    # BS x T
                self.bert_tbl_mask      = tf.placeholder( tf.int32, name = "bert_tbl_mask", shape = [ None ] )
                self.bert_tbl_col_idx   = tf.reshape( self.bert_tbl_col_idx, [ tf.shape( self.bert_tbl_col_mask )[0], tf.shape( self.bert_tbl_col_mask )[1], -1 ] )

                batch_size          = tf.shape( self.bert_idx )[0]

                ###########################################
                # BINARY CLASSIFICATION: PATH INFORMATION #
                ###########################################
                self.cur_path_main  = tf.placeholder( tf.bool, name = "cur_path_main", shape = [ None ] )      # BS. 1 if the current path is main; 0 otherwise.
                self.cur_path_idx   = tf.placeholder( tf.int32, name = PF_PATHIDX, shape = [ None, None ] ) # BS X P. Current Path. P = Path Depth.
                self.cur_path_mask  = tf.placeholder( tf.int32, name = "cur_path_mask", shape = [ None ] )      # BS. 

                ################################
                # TABLE CLASSIFICATION TARGETS #
                ################################
                self.c_tbl_num      = tf.placeholder( tf.int32, name = TV_TABLES_NUM, shape = [ None ] )          # Used table numbers. MAX NUM: 5
                self.c_tables       = tf.placeholder( tf.int32, name = TV_TABLES_USED_IDX, shape = [ None, None ] )     # BS x B. One-hot vectors.

                ##################################
                # BINARY CLASSIFICATION: TARGETS #
                ##################################
                # BELOW TWO are valid only when the path is [ NONE ].
                self.c_merge_op     = tf.placeholder( tf.int32, name = PF_MERGEOP, shape = [ None ] )         # BS. Merge Operation. 4 Classes.
                self.c_from_sql     = tf.placeholder( tf.int32, name = PF_FROMSQL, shape = [ None ] )         # BS. 1 if True. "From" part of the query is SQL statement, not a tbl.

                # BELOW FIVE are valid for all pathes.
                self.c_has_orderby  = tf.placeholder( tf.int32, name = PF_ORDERBY, shape = [ None ] )
                self.c_has_groupby  = tf.placeholder( tf.int32, name = PF_GROUPBY, shape = [ None ] )
                self.c_has_limit    = tf.placeholder( tf.int32, name = PF_LIMIT, shape = [ None ] )
                self.c_has_where    = tf.placeholder( tf.int32, name = PF_WHERE, shape = [ None ] )
                self.c_has_having   = tf.placeholder( tf.int32, name = PF_HAVING, shape = [ None ] )
                
                ####################################
                # GROUPBY COMPONENT CLASSIFICATION #
                ####################################
                self.c_gb_num       = tf.placeholder_with_default( tf.zeros( [ batch_size ], tf.int32 ), name = GF_NUMCOL, shape = [ None ] )          # Groupby column numbers. MAX 3.
                self.c_gb_col       = tf.placeholder( tf.int32, name = GF_COLLIST, shape = [ None, None ] )   # BS X P. One-hot vectors.
                
                ####################################
                # ORDERBY COMPONENT CLASSIFICATION #
                ####################################
                self.c_ob_num       = tf.placeholder_with_default( tf.zeros( [ batch_size ], tf.int32 ), name = OF_NUMVU, shape = [ None ] )          # Orderby column numbers. MAX 3
                self.c_ob_desc      = tf.placeholder( tf.int32, name = OF_DESCRIPTOR, shape = [ None ] )         # Orderby Descriptor.
                self.c_ob_p1        = tf.placeholder( tf.int32, name = OF_VU_COL1, shape = [ None, None ] )
                self.c_ob_p2        = tf.placeholder( tf.int32, name = OF_VU_COL2, shape = [ None, None ] )
                self.c_ob_agg1      = tf.placeholder( tf.int32, name = OF_VU_AGG1, shape = [ None, None ] )
                self.c_ob_agg2      = tf.placeholder( tf.int32, name = OF_VU_AGG2, shape = [ None, None ] )
                self.c_ob_dist1     = tf.placeholder( tf.int32, name = OF_VU_DIST1, shape = [ None, None ] )
                self.c_ob_dist2     = tf.placeholder( tf.int32, name = OF_VU_DIST2, shape = [ None, None ] )
                self.c_ob_op        = tf.placeholder( tf.int32, name = OF_VU_OPERATOR, shape = [ None, None ] )     # BS X 3.

                ##################################
                # LIMIT COMPONENT CLASSIFICATION #
                ##################################
                self.c_lm_ismax     = tf.placeholder( tf.int32, name = LF_ISMAX, shape = [ None ] )
                self.c_lm_maxptr    = tf.placeholder( tf.int32, name = LF_POINTERLOC, shape = [ None ] )

                ###################################
                # SELECT COMPONENT CLASSIFICATION #
                ###################################
                self.c_sel_num      = tf.placeholder_with_default( tf.zeros( [ batch_size ], tf.int32 ), name = SF_NUM_VU, shape = [ None ] )       # Select column numbers. MAX 6
                self.c_sel_dist     = tf.placeholder( tf.int32, name = SF_DISTINCT, shape = [ None ] )  
                self.c_sel_aggtot   = tf.placeholder( tf.int32, name = SF_VU_AGGALL, shape = [ None, None ] )
                self.c_sel_p1       = tf.placeholder( tf.int32, name = SF_VU_COL1, shape = [ None, None ] )
                self.c_sel_p2       = tf.placeholder( tf.int32, name = SF_VU_COL2, shape = [ None, None ] )
                self.c_sel_agg1     = tf.placeholder( tf.int32, name = SF_VU_AGG1, shape = [ None, None ] )
                self.c_sel_agg2     = tf.placeholder( tf.int32, name = SF_VU_AGG2, shape = [ None, None ] )
                self.c_sel_dist1    = tf.placeholder( tf.int32, name = SF_VU_DIST1, shape = [ None, None ] )
                self.c_sel_dist2    = tf.placeholder( tf.int32, name = SF_VU_DIST2, shape = [ None, None ] )
                self.c_sel_op       = tf.placeholder( tf.int32, name = SF_VU_OPERATOR, shape = [ None, None ] )     # BS X 3.

                ##################################
                # WHERE COMPONENT CLASSIFICATION #
                ##################################
                self.c_where_num    = tf.placeholder_with_default( tf.zeros( [ batch_size ], tf.int32 ), name = WF_NUM_CONDUNIT, shape = [ None ] )        # Where column number. MAX 4
                self.c_wh_agg       = tf.placeholder( tf.int32, name = WF_CU_AGGREGATOR, shape = [ None, None ] )
                self.c_wh_isnot     = tf.placeholder( tf.int32, name = WF_CU_IS_NOT, shape = [ None, None ] )
                self.c_wh_condop    = tf.placeholder( tf.int32, name = WF_CU_COND_OP, shape = [ None, None ] )
                self.c_wh_val1_type = tf.placeholder( tf.int32, name = WF_CU_VAL1_TYPE, shape = [ None, None ] )
                self.c_wh_val1_sp   = tf.placeholder( tf.int32, name = WF_CU_VAL1_SP, shape = [ None, None ] )
                self.c_wh_val1_ep   = tf.placeholder( tf.int32, name = WF_CU_VAL1_EP, shape = [ None, None ] )
                self.c_wh_val1_like = tf.placeholder( tf.int32, name = WF_CU_VAL1_LIKELY, shape = [ None, None ] )
                self.c_wh_val1_bool = tf.placeholder( tf.int32, name = WF_CU_VAL1_BOOLVAL, shape = [ None, None ] )
                self.c_wh_val2_type = tf.placeholder( tf.int32, name = WF_CU_VAL2_TYPE, shape = [ None, None ] )
                self.c_wh_val2_sp   = tf.placeholder( tf.int32, name = WF_CU_VAL2_SP, shape = [ None, None ] )
                self.c_wh_val2_ep   = tf.placeholder( tf.int32, name = WF_CU_VAL2_EP, shape = [ None, None ] )
                self.c_wh_val2_like = tf.placeholder( tf.int32, name = WF_CU_VAL2_LIKELY, shape = [ None, None ] )
                self.c_wh_val2_bool = tf.placeholder( tf.int32, name = WF_CU_VAL2_BOOLVAL, shape = [ None, None ] )
                self.c_wh_vu_op     = tf.placeholder( tf.int32, name = WF_CU_VU_OPERATOR, shape = [ None, None ] )
                self.c_wh_vu_agg1   = tf.placeholder( tf.int32, name = WF_CU_VU_AGG1, shape = [ None, None ] )
                self.c_wh_vu_col1   = tf.placeholder( tf.int32, name = WF_CU_VU_COL1, shape = [ None, None ] )
                self.c_wh_vu_dist1  = tf.placeholder( tf.int32, name = WF_CU_VU_DIST1, shape = [ None, None ] )
                self.c_wh_vu_agg2   = tf.placeholder( tf.int32, name = WF_CU_VU_AGG2, shape = [ None, None ] )
                self.c_wh_vu_col2   = tf.placeholder( tf.int32, name = WF_CU_VU_COL2, shape = [ None, None ] )
                self.c_wh_vu_dist2  = tf.placeholder( tf.int32, name = WF_CU_VU_DIST2, shape = [ None, None ] )

                self.meta_wh_igval1 = tf.placeholder( tf.int32, name = WF_CU_VAL1_IGNORE, shape = [ None, None ] )
                self.meta_wh_igval2 = tf.placeholder( tf.int32, name = WF_CU_VAL2_IGNORE, shape = [ None, None ] )
                
                ###################################
                # HAVING COMPONENT CLASSIFICATION #
                ###################################
                self.c_having_num   = tf.placeholder_with_default( tf.zeros( [ batch_size ], tf.int32 ), name = HV_NUM_CONDUNIT, shape = [ None ] )        # Where column number. MAX 4
                self.c_hv_agg       = tf.placeholder( tf.int32, name = HV_CU_AGGREGATOR, shape = [ None, None ] )
                self.c_hv_isnot     = tf.placeholder( tf.int32, name = HV_CU_IS_NOT, shape = [ None, None ] )
                self.c_hv_condop    = tf.placeholder( tf.int32, name = HV_CU_COND_OP, shape = [ None, None ] )
                self.c_hv_val1_type = tf.placeholder( tf.int32, name = HV_CU_VAL1_TYPE, shape = [ None, None ] )
                self.c_hv_val1_sp   = tf.placeholder( tf.int32, name = HV_CU_VAL1_SP, shape = [ None, None ] )
                self.c_hv_val1_ep   = tf.placeholder( tf.int32, name = HV_CU_VAL1_EP, shape = [ None, None ] )
                self.c_hv_val1_like = tf.placeholder( tf.int32, name = HV_CU_VAL1_LIKELY, shape = [ None, None ] )
                self.c_hv_val1_bool = tf.placeholder( tf.int32, name = HV_CU_VAL1_BOOLVAL, shape = [ None, None ] )
                self.c_hv_val2_type = tf.placeholder( tf.int32, name = HV_CU_VAL2_TYPE, shape = [ None, None ] )
                self.c_hv_val2_sp   = tf.placeholder( tf.int32, name = HV_CU_VAL2_SP, shape = [ None, None ] )
                self.c_hv_val2_ep   = tf.placeholder( tf.int32, name = HV_CU_VAL2_EP, shape = [ None, None ] )
                self.c_hv_val2_like = tf.placeholder( tf.int32, name = HV_CU_VAL2_LIKELY, shape = [ None, None ] )
                self.c_hv_val2_bool = tf.placeholder( tf.int32, name = HV_CU_VAL2_BOOLVAL, shape = [ None, None ] )
                self.c_hv_vu_op     = tf.placeholder( tf.int32, name = HV_CU_VU_OPERATOR, shape = [ None, None ] )
                self.c_hv_vu_agg1   = tf.placeholder( tf.int32, name = HV_CU_VU_AGG1, shape = [ None, None ] )
                self.c_hv_vu_col1   = tf.placeholder( tf.int32, name = HV_CU_VU_COL1, shape = [ None, None ] )
                self.c_hv_vu_dist1  = tf.placeholder( tf.int32, name = HV_CU_VU_DIST1, shape = [ None, None ] )
                self.c_hv_vu_agg2   = tf.placeholder( tf.int32, name = HV_CU_VU_AGG2, shape = [ None, None ] )
                self.c_hv_vu_col2   = tf.placeholder( tf.int32, name = HV_CU_VU_COL2, shape = [ None, None ] )
                self.c_hv_vu_dist2  = tf.placeholder( tf.int32, name = HV_CU_VU_DIST2, shape = [ None, None ] )

                self.meta_hv_igval1 = tf.placeholder( tf.int32, name = HV_CU_VAL1_IGNORE, shape = [ None, None ] )
                self.meta_hv_igval2 = tf.placeholder( tf.int32, name = HV_CU_VAL2_IGNORE, shape = [ None, None ] )
                
                ###########################
                # OTHER LEARNING METADATA #
                ###########################
                self.is_train       = tf.placeholder( tf.bool, name = "is_train", shape = [] )
                self.drop_rate      = tf.placeholder( tf.float32, name = "drop_rate", shape = [] )
                self.global_step    = tf.placeholder( tf.float32, name = "global_step", shape = [] )
                regularizer         = tf.contrib.layers.l2_regularizer( scale = 1e-5 )

                keep_prob           = 1.0 - self.drop_rate

                # 1. Encode current path.
                path_embed  = self.pe.get_path_embeddings( 100, \
                                        self.cur_path_idx, self.cur_path_mask, \
                                        150, "PE", \
                                        self.is_train, 1.0 - self.drop_rate, regularizer = regularizer )    # BS X ( 8 * 128 )
           
                q_tok_embed, col_embed, q_embed, tbl_embed  = self.je.get_joint_embedding( \
                                                                    self.bert_idx, self.bert_mask, self.bert_seg_id, \
                                                                    self.bert_q_mask, self.bert_col_loc, self.bert_col_mask, \
                                                                    self.is_train, "JE", 1.0 - self.drop_rate, regularizer = regularizer )

                q_db_embed, db_tbl_embed    = self.je.get_tbl_joint_embedding( \
                                                                    self.bert_idx, self.bert_mask, self.bert_seg_id, \
                                                                    self.bert_q_mask, self.bert_col_loc, self.bert_col_mask, \
                                                                    self.bert_tbl_col_idx, self.bert_tbl_col_mask, self.bert_tbl_mask, \
                                                                    self.is_train, "JE", 1.0 - self.drop_rate, regularizer = regularizer )

                integrated_info = tf.concat( [ tbl_embed, path_embed ], -1 )
                integrated_info = tf.nn.dropout( integrated_info, rate = self.drop_rate )

                integrated_info = tf.layers.dense( integrated_info, 256, kernel_initializer = variable_initializer, name = "full", kernel_regularizer = regularizer, activation = tf.nn.leaky_relu )
                integrated_info = tf.contrib.layers.layer_norm( integrated_info, scope = "ln_integrated" )
                integrated_info = tf.nn.dropout( integrated_info, rate = self.drop_rate )

                # 4. Table Selection Network.
                with tf.variable_scope( "Table_Classification" ):
                    att_dim         = q_db_embed.get_shape().as_list()[-1]

                    # Whole database representation.
                    db_as_whole     = tf.squeeze( self.je._self_att( db_tbl_embed, self.bert_tbl_mask, keep_prob, regularizer, scope = "db_emb" ), 1 )
                    tbl_scores      = tf.tanh( tf.layers.dense( db_tbl_embed, att_dim, kernel_initializer = variable_initializer, name = "tbl_att1", kernel_regularizer = regularizer, use_bias = False ) + \
                                                tf.layers.dense( tf.expand_dims( q_db_embed, 1 ), att_dim, kernel_initializer = variable_initializer, name = "tbl_att2", kernel_regularizer = regularizer, use_bias = False ) + \
                                                tf.layers.dense( tf.expand_dims( db_as_whole, 1 ), att_dim, kernel_initializer = variable_initializer, name = "tbl_att3", kernel_regularizer = regularizer, use_bias = False ) + \
                                                tf.layers.dense( tf.expand_dims( path_embed, 1 ), att_dim, kernel_initializer = variable_initializer, name = "tbl_att4", kernel_regularizer = regularizer, use_bias = False ) )
                    tbl_scores      = tf.layers.dense( tbl_scores, 1, kernel_initializer = variable_initializer, name = "tbl_score", kernel_regularizer = regularizer, use_bias = False ) # BS X B X 1.
                    tbl_scores      = tf.transpose( apply_mask( tbl_scores, self.bert_tbl_mask, float( "-inf" ) ), [ 0, 2, 1 ] ) # BS X 1 X B
                    self.tables_s   = tf.squeeze( tbl_scores , 1 )    # BS X B.
                    tbl_self_att    = tf.nn.softmax( tbl_scores, -1 )
                    updated_context = tf.matmul( tbl_self_att, db_tbl_embed )   # BS X 1 X D.
                    
                    self.tbl_num_r, self.tbl_num_s  = self._prop_class_net( tf.concat( [ q_db_embed, db_as_whole, tf.squeeze( updated_context, 1 ) ], -1 ), 6, "TBL_NUM", keep_prob, regularizer )

                    prop_table_loss = 0.0
                    prop_table_loss     += tf.reduce_mean( tf.nn.sparse_softmax_cross_entropy_with_logits( logits = self.tbl_num_s, labels = self.c_tbl_num ) )
                    tbl_selection_loss  = tf.nn.sigmoid_cross_entropy_with_logits( logits = self.tables_s, labels = tf.cast( self.c_tables, dtype = tf.float32 ) )  # BS X B.
                    tbl_mask            = tf.sequence_mask( self.bert_tbl_mask, tf.shape( self.c_tables )[1] )
                    tbl_selection_loss  = tf.where( tbl_mask, tbl_selection_loss, tf.zeros_like( tbl_selection_loss ) )
                    prop_table_loss     += tf.reduce_sum( tbl_selection_loss ) / tf.cast( tf.reduce_sum( tf.cast( tbl_mask, tf.int32 ) ), tf.float32 )

                # 5. Property Classification Network.
                with tf.variable_scope( "Property_Classification" ):
                    self.merge_op_result, self.merge_op_score   = self._prop_class_net( integrated_info, 4, "MERGE_OP", keep_prob, regularizer )
                    self.from_sql_result, self.from_sql_score   = self._prop_class_net( integrated_info, 2, "FROM_SQL", keep_prob, regularizer )
                    self.order_by_result, self.order_by_score   = self._prop_class_net( integrated_info, 2, "ORDER_BY", keep_prob, regularizer )
                    self.group_by_result, self.group_by_score   = self._prop_class_net( integrated_info, 2, "GROUP_BY", keep_prob, regularizer )
                    self.limit_result, self.limit_score         = self._prop_class_net( integrated_info, 2, "LIMIT", keep_prob, regularizer )
                    self.where_result, self.where_score         = self._prop_class_net( integrated_info, 2, "WHERE", keep_prob, regularizer )
                    self.having_result, self.having_score       = self._prop_class_net( integrated_info, 2, "HAVING", keep_prob, regularizer )

                    # Property Classification: LOSS Calculation.
                    prop_class_loss = 0.0
                    prop_class_loss += self._get_conditioned_loss( self.cur_path_main, self.merge_op_score, self.c_merge_op )
                    prop_class_loss += self._get_conditioned_loss( self.cur_path_main, self.from_sql_score, self.c_from_sql )
                    prop_class_loss += tf.reduce_mean( tf.nn.sparse_softmax_cross_entropy_with_logits( logits = self.order_by_score, labels = self.c_has_orderby ) )
                    prop_class_loss += tf.reduce_mean( tf.nn.sparse_softmax_cross_entropy_with_logits( logits = self.group_by_score, labels = self.c_has_groupby ) )
                    prop_class_loss += tf.reduce_mean( tf.nn.sparse_softmax_cross_entropy_with_logits( logits = self.limit_score, labels = self.c_has_limit ) )
                    prop_class_loss += tf.reduce_mean( tf.nn.sparse_softmax_cross_entropy_with_logits( logits = self.where_score, labels = self.c_has_where ) )
                    prop_class_loss += tf.reduce_mean( tf.nn.sparse_softmax_cross_entropy_with_logits( logits = self.having_score, labels = self.c_has_having ) )

                dim         = q_tok_embed.get_shape().as_list()[-1]

                # 6. GroupBy Network.
                with tf.variable_scope( "GroupBy_Classification" ):
                    self.gb_num_result, self.gb_num_score   = self._prop_class_net( integrated_info, 4, "GB_NUM", keep_prob, regularizer )

                    path_dim    = path_embed.get_shape().as_list()[-1]
                    qtok_self   = tf.tanh( tf.layers.dense( q_tok_embed, path_dim, kernel_initializer = variable_initializer, name = "att_q", kernel_regularizer = regularizer, use_bias = False ) + \
                                            tf.layers.dense( tf.expand_dims( path_embed, 1 ), path_dim, kernel_initializer = variable_initializer, name = "att_path", kernel_regularizer = regularizer, use_bias = False ) )
                    qtok_self   = tf.layers.dense( qtok_self, 3, kernel_initializer = variable_initializer, name = "qtok_a", kernel_regularizer = regularizer, use_bias = False )
                    qtok_self   = apply_mask( qtok_self, self.bert_q_mask, float( "-inf" ) )
                    qtok_self   = tf.nn.softmax( tf.transpose( qtok_self, [ 0, 2, 1 ] ), -1 )
                    qtok_summary    = tf.matmul( qtok_self, q_tok_embed )   # BS X 3 X D
                    
                    att_mask    = tf.cond( tf.equal( self.is_train, True ), lambda: self.c_gb_num, lambda: tf.cast( self.gb_num_result, tf.int32 ) )

                    # Since we are directly calling the pointer network...
                    qtok_summary    = apply_mask( qtok_summary, att_mask, float( 0.0 ) )
                    qtok_summary    = layer_norm( qtok_summary, att_mask, scope = "c_ln" )
                    mc, p           = get_pointer_N( qtok_summary, col_embed, self.bert_col_mask, regularizer, scope = "PTR1", dim_scale = 2 )
                    self.gb_col_result  = p

                    gb_col_key      = tf.sequence_mask( self.c_gb_num, 3 )  # BS X 3.

                    prop_gb_loss    = 0.0
                    prop_gb_loss    += self._get_conditioned_loss( self.c_has_groupby, self.gb_num_score, self.c_gb_num )
                    prop_gb_loss    += self._get_conditioned_loss( gb_col_key, p, self.c_gb_col )

                # 7. OrderBy Network
                with tf.variable_scope( "OrderBy_Classification" ):
                    self.ob_num_result, self.ob_num_score   = self._prop_class_net( integrated_info, 4, "OB_NUM", keep_prob, regularizer )
                    
                    # Question summary, based on PATH.
                    path_dim    = path_embed.get_shape().as_list()[-1]
                    qtok_self   = tf.tanh( tf.layers.dense( q_tok_embed, path_dim, kernel_initializer = variable_initializer, name = "att_q", kernel_regularizer = regularizer, use_bias = False ) + \
                                            tf.layers.dense( tf.expand_dims( path_embed, 1 ), path_dim, kernel_initializer = variable_initializer, name = "att_path", kernel_regularizer = regularizer, use_bias = False ) )
                    qtok_self   = tf.layers.dense( qtok_self, 3, kernel_initializer = variable_initializer, name = "qtok_a", kernel_regularizer = regularizer, use_bias = False )
                    qtok_self   = apply_mask( qtok_self, self.bert_q_mask, float( "-inf" ) )
                    qtok_self   = tf.nn.softmax( tf.transpose( qtok_self, [ 0, 2, 1 ] ), -1 )
                    qtok_summary    = tf.matmul( qtok_self, q_tok_embed )   # BS X 3 X D
                    
                    att_mask    = tf.cond( tf.equal( self.is_train, True ), lambda: self.c_ob_num, lambda: tf.cast( self.ob_num_result, tf.int32 ) )

                    mc1, _, self.ob_p1_r, self.ob_p2_r, self.ob_agg1_r, self.ob_agg2_r, self.ob_dist1_r, self.ob_dist2_r, self.ob_op_r  = generate_multiple_vu( qtok_summary, att_mask, col_embed, self.bert_col_mask, 3, "OB_VU", regularizer )    # BS X 3 X T for ALL.
                    self.ob_desc_result, self.ob_desc_score = self._prop_class_net( mc1[:,0,:], 2, "OB_DESC", keep_prob, regularizer )

                    ob_col_key      = tf.sequence_mask( self.c_ob_num, 3 )  # BS X 3.
                    ob_col_key2     = tf.not_equal( self.c_ob_op, 0 )      # BS X 3. True: Evaluate on COL2. False: Otherwise.
                    ob_col_key2     = tf.logical_and( ob_col_key, ob_col_key2 )

                    prop_ob_loss    = 0.0
                    prop_ob_loss    += self._get_conditioned_loss( self.c_has_orderby, self.ob_num_score, self.c_ob_num )
                    prop_ob_loss    += self._get_conditioned_loss( self.c_has_orderby, self.ob_desc_score, self.c_ob_desc )
                    prop_ob_loss    += self._get_conditioned_loss( ob_col_key, self.ob_p1_r, self.c_ob_p1 )
                    prop_ob_loss    += self._get_conditioned_loss( ob_col_key, self.ob_agg1_r, self.c_ob_agg1 )
                    prop_ob_loss    += self._get_conditioned_loss( ob_col_key, self.ob_dist1_r, self.c_ob_dist1 )
                    prop_ob_loss    += self._get_conditioned_loss( ob_col_key2, self.ob_p2_r, self.c_ob_p2 )
                    prop_ob_loss    += self._get_conditioned_loss( ob_col_key2, self.ob_agg2_r, self.c_ob_agg2 )
                    prop_ob_loss    += self._get_conditioned_loss( ob_col_key2, self.ob_dist2_r, self.c_ob_dist2 )
                    prop_ob_loss    += self._get_conditioned_loss( ob_col_key, self.ob_op_r, self.c_ob_op )

                # 8. Limit Network.
                with tf.variable_scope( "LIMIT_CLASSIFICATION" ):
                    # Question Pointer.
                    path_dim    = path_embed.get_shape().as_list()[-1]
                    qtok_self   = tf.tanh( tf.layers.dense( q_tok_embed, path_dim, kernel_initializer = variable_initializer, name = "att_q", kernel_regularizer = regularizer, use_bias = False ) + \
                                            tf.layers.dense( tf.expand_dims( path_embed, 1 ), path_dim, kernel_initializer = variable_initializer, name = "att_path", kernel_regularizer = regularizer, use_bias = False ) )
                    qtok_self   = tf.layers.dense( qtok_self, 1, kernel_initializer = variable_initializer, name = "qtok_a", kernel_regularizer = regularizer, use_bias = False )
                    qtok_self   = apply_mask( qtok_self, self.bert_q_mask, float( "-inf" ) )
                    qtok_self   = tf.nn.softmax( tf.transpose( qtok_self, [ 0, 2, 1 ] ), -1 )   # BS X 1 X T.

                    self.lm_ismax_r, self.lm_ismax_score    = self._prop_class_net( integrated_info, 2, "LIMIT_ISMAX", keep_prob, regularizer )
                    self.lm_maxptr_score    = tf.squeeze( qtok_self, 1 )
                    self.lm_maxptr_r        = tf.argmax( self.lm_maxptr_score, -1 )

                    # Limit: Loss calculation.
                    prop_limit_loss = 0.0
                    prop_limit_loss += self._get_conditioned_loss( self.c_has_limit, self.lm_ismax_score, self.c_lm_ismax )
                    prop_limit_loss += self._get_conditioned_loss( tf.logical_and( tf.cast( self.c_has_limit, tf.bool ), tf.logical_not( tf.cast( self.c_lm_ismax, tf.bool ) ) ), self.lm_maxptr_score, self.c_lm_maxptr )
    
                # 9. Select Network
                with tf.variable_scope( "Select_Classification" ):
                    self.sel_num_r, self.sel_num_score      = self._prop_class_net( integrated_info, 7, "SEL_NUM", keep_prob, regularizer )
                    self.sel_dist_r, self.sel_dist_score    = self._prop_class_net( integrated_info, 2, "SEL_DIST", keep_prob, regularizer )
                    
                    # Question summary.
                    path_dim    = path_embed.get_shape().as_list()[-1]
                    qtok_self   = tf.tanh( tf.layers.dense( q_tok_embed, path_dim, kernel_initializer = variable_initializer, name = "att_q", kernel_regularizer = regularizer, use_bias = False ) + \
                                            tf.layers.dense( tf.expand_dims( path_embed, 1 ), path_dim, kernel_initializer = variable_initializer, name = "att_path", kernel_regularizer = regularizer, use_bias = False ) )
                    qtok_self   = tf.layers.dense( qtok_self, 6, kernel_initializer = variable_initializer, name = "qtok_a", kernel_regularizer = regularizer, use_bias = False )
                    qtok_self   = apply_mask( qtok_self, self.bert_q_mask, float( "-inf" ) )
                    qtok_self   = tf.nn.softmax( tf.transpose( qtok_self, [ 0, 2, 1 ] ), -1 )
                    qtok_summary    = tf.matmul( qtok_self, q_tok_embed )   # BS X 6 X D

                    
                    att_mask    = tf.cond( tf.equal( self.is_train, True ), lambda: self.c_sel_num, lambda: tf.cast( self.sel_num_r, tf.int32 ) )

                    _, _, self.sel_p1_r, self.sel_p2_r, self.sel_agg1_r, self.sel_agg2_r, self.sel_dist1_r, self.sel_dist2_r, self.sel_op_r, self.sel_aggtot_r  = generate_multiple_vu( qtok_summary, att_mask, col_embed, self.bert_col_mask, 6, "SEL_VU", regularizer, do_agg_tot = True )    # BS X 6 X T for ALL.

                    sel_col_key     = tf.sequence_mask( self.c_sel_num, 6 )  # BS X 5
                    sel_col_key2    = tf.not_equal( self.c_sel_op, 0 )      # BS X 5. True: Evaluate on COL2. False: Otherwise.
                    sel_col_key2    = tf.logical_and( sel_col_key, sel_col_key2 )

                    prop_sel_loss   = 0.0
                    prop_sel_loss   += tf.reduce_mean( tf.nn.sparse_softmax_cross_entropy_with_logits( logits = self.sel_num_score, labels = self.c_sel_num ) )
                    prop_sel_loss   += tf.reduce_mean( tf.nn.sparse_softmax_cross_entropy_with_logits( logits = self.sel_dist_score, labels = self.c_sel_dist ) )
                    prop_sel_loss   += self._get_conditioned_loss( sel_col_key, self.sel_aggtot_r, self.c_sel_aggtot )
                    prop_sel_loss   += self._get_conditioned_loss( sel_col_key, self.sel_p1_r, self.c_sel_p1 )
                    prop_sel_loss   += self._get_conditioned_loss( sel_col_key, self.sel_agg1_r, self.c_sel_agg1 )
                    prop_sel_loss   += self._get_conditioned_loss( sel_col_key, self.sel_dist1_r, self.c_sel_dist1 )
                    prop_sel_loss   += self._get_conditioned_loss( sel_col_key2, self.sel_p2_r, self.c_sel_p2 )
                    prop_sel_loss   += self._get_conditioned_loss( sel_col_key2, self.sel_agg2_r, self.c_sel_agg2 )
                    prop_sel_loss   += self._get_conditioned_loss( sel_col_key2, self.sel_dist2_r, self.c_sel_dist2 )
                    prop_sel_loss   += self._get_conditioned_loss( sel_col_key, self.sel_op_r, self.c_sel_op )

                # 10. Where network.
                with tf.variable_scope( "Where_Classification" ):
                    self.where_num_r, self.where_num_score  = self._prop_class_net( integrated_info, 5, "WHERE_NUM", keep_prob, regularizer )       # CLASSES: 0, 1, 2, 3, 4 

                    # Question Summary.
                    path_dim    = path_embed.get_shape().as_list()[-1]
                    qtok_self   = tf.tanh( tf.layers.dense( q_tok_embed, path_dim, kernel_initializer = variable_initializer, name = "att_q", kernel_regularizer = regularizer, use_bias = False ) + \
                                            tf.layers.dense( tf.expand_dims( path_embed, 1 ), path_dim, kernel_initializer = variable_initializer, name = "att_path", kernel_regularizer = regularizer, use_bias = False ) )
                    qtok_self   = tf.layers.dense( qtok_self, 4, kernel_initializer = variable_initializer, name = "qtok_a", kernel_regularizer = regularizer, use_bias = False )
                    qtok_self   = apply_mask( qtok_self, self.bert_q_mask, float( "-inf" ) )
                    qtok_self   = tf.nn.softmax( tf.transpose( qtok_self, [ 0, 2, 1 ] ), -1 )
                    qtok_summary    = tf.matmul( qtok_self, q_tok_embed )
                    
                    att_mask    = tf.cond( tf.equal( self.is_train, True ), lambda: self.c_where_num, lambda: tf.cast( self.where_num_r, tf.int32 ) )

                    self.wh_agg_s, self.wh_isnot_s, self.wh_condop_s, \
                    self.wh_val1_type_s, self.wh_val1_like_s, self.wh_val1_bool_s, self.wh_val1_sp_s, self.wh_val1_ep_s, \
                    self.wh_val2_type_s, self.wh_val2_like_s, self.wh_val2_bool_s, self.wh_val2_sp_s, self.wh_val2_ep_s, \
                    self.wh_vu_col1_s, self.wh_vu_col2_s, self.wh_vu_agg1_s, self.wh_vu_agg2_s, self.wh_vu_dist1_s, self.wh_vu_dist2_s, self.wh_vu_op_s = generate_multiple_cu( qtok_summary, att_mask, q_tok_embed, col_embed, self.bert_q_mask, self.bert_col_mask, 4, "WHERE_CU", regularizer )

                    # Loss calculation.
                    prop_where_loss = 0.0
                    base_mask       = tf.logical_and( tf.sequence_mask( self.c_where_num, 4 ), tf.tile( tf.expand_dims( tf.cast( self.c_has_where, tf.bool ), -1 ), [ 1, 4 ] ) )
                    val1_se_mask    = tf.logical_and( base_mask, tf.equal( self.c_wh_val1_type, 0 ) )
                    val1_se_mask    = tf.logical_and( val1_se_mask, tf.logical_not( tf.cast( self.meta_wh_igval1, tf.bool ) ) )

                    val1_b_mask     = tf.logical_and( base_mask, tf.equal( self.c_wh_val1_type, 1 ) )
                    val1_b_mask     = tf.logical_and( val1_b_mask, tf.logical_not( tf.cast( self.meta_wh_igval1, tf.bool ) ) )

                    val2_mask       = tf.logical_and( base_mask, tf.equal( self.c_wh_condop, 0 ) )      # ONLY between case.
                    val2_se_mask    = tf.logical_and( val2_mask, tf.equal( self.c_wh_val2_type, 0 ) )
                    val2_se_mask    = tf.logical_and( val2_se_mask, tf.logical_not( tf.cast( self.meta_wh_igval2, tf.bool ) ) )

                    val2_b_mask     = tf.logical_and( val2_mask, tf.equal( self.c_wh_val2_type, 1 ) )
                    val2_b_mask     = tf.logical_and( val2_b_mask, tf.logical_not( tf.cast( self.meta_wh_igval2, tf.bool ) ) )

                    vu_val2_mask    = tf.logical_and( base_mask, tf.not_equal( self.c_wh_vu_op, 0 ) )
                    prop_where_loss += self._get_conditioned_loss( self.c_has_where, self.where_num_score, self.c_where_num )
                    prop_where_loss += self._get_conditioned_loss( base_mask, self.wh_agg_s, self.c_wh_agg )
                    prop_where_loss += self._get_conditioned_loss( base_mask, self.wh_isnot_s, self.c_wh_isnot )
                    prop_where_loss += self._get_conditioned_loss( base_mask, self.wh_condop_s, self.c_wh_condop )
                    prop_where_loss += self._get_conditioned_loss( base_mask, self.wh_val1_type_s, self.c_wh_val1_type )
                    prop_where_loss += self._get_conditioned_loss( val1_se_mask, self.wh_val1_like_s, self.c_wh_val1_like ) # Only when the text match is text span.
                    prop_where_loss += self._get_conditioned_loss( val1_b_mask, self.wh_val1_bool_s, self.c_wh_val1_bool )
                    prop_where_loss += self._get_conditioned_loss( val1_se_mask, self.wh_val1_sp_s, self.c_wh_val1_sp )
                    prop_where_loss += self._get_conditioned_loss( val1_se_mask, self.wh_val1_ep_s, self.c_wh_val1_ep )
                    prop_where_loss += self._get_conditioned_loss( val2_mask, self.wh_val2_type_s, self.c_wh_val2_type )
                    prop_where_loss += self._get_conditioned_loss( val2_se_mask, self.wh_val2_like_s, self.c_wh_val2_like )
                    prop_where_loss += self._get_conditioned_loss( val2_b_mask, self.wh_val2_bool_s, self.c_wh_val2_bool )
                    prop_where_loss += self._get_conditioned_loss( val2_se_mask, self.wh_val2_sp_s, self.c_wh_val2_sp )
                    prop_where_loss += self._get_conditioned_loss( val2_se_mask, self.wh_val2_ep_s, self.c_wh_val2_ep )
                    prop_where_loss += self._get_conditioned_loss( base_mask, self.wh_vu_op_s, self.c_wh_vu_op )
                    prop_where_loss += self._get_conditioned_loss( base_mask, self.wh_vu_agg1_s, self.c_wh_vu_agg1 )
                    prop_where_loss += self._get_conditioned_loss( base_mask, self.wh_vu_col1_s, self.c_wh_vu_col1 )
                    prop_where_loss += self._get_conditioned_loss( base_mask, self.wh_vu_dist1_s, self.c_wh_vu_dist1 )
                    prop_where_loss += self._get_conditioned_loss( vu_val2_mask, self.wh_vu_agg2_s, self.c_wh_vu_agg2 )
                    prop_where_loss += self._get_conditioned_loss( vu_val2_mask, self.wh_vu_col2_s, self.c_wh_vu_col2 )
                    prop_where_loss += self._get_conditioned_loss( vu_val2_mask, self.wh_vu_dist2_s, self.c_wh_vu_dist2 )

                # 11. Having network.
                with tf.variable_scope( "Having_Classification" ):
                    self.having_num_r, self.having_num_score  = self._prop_class_net( integrated_info, 3, "HAVING_NUM", keep_prob, regularizer )       # CLASSES: 0, 1, 2

                    # Question Summary.
                    path_dim    = path_embed.get_shape().as_list()[-1]
                    qtok_self   = tf.tanh( tf.layers.dense( q_tok_embed, path_dim, kernel_initializer = variable_initializer, name = "att_q", kernel_regularizer = regularizer, use_bias = False ) + \
                                            tf.layers.dense( tf.expand_dims( path_embed, 1 ), path_dim, kernel_initializer = variable_initializer, name = "att_path", kernel_regularizer = regularizer, use_bias = False ) )
                    qtok_self   = tf.layers.dense( qtok_self, 2, kernel_initializer = variable_initializer, name = "qtok_a", kernel_regularizer = regularizer, use_bias = False )
                    qtok_self   = apply_mask( qtok_self, self.bert_q_mask, float( "-inf" ) )
                    qtok_self   = tf.nn.softmax( tf.transpose( qtok_self, [ 0, 2, 1 ] ), -1 )
                    qtok_summary    = tf.matmul( qtok_self, q_tok_embed )
                    
                    att_mask    = tf.cond( tf.equal( self.is_train, True ), lambda: self.c_having_num, lambda: tf.cast( self.having_num_r, tf.int32 ) )

                    self.hv_agg_s, self.hv_isnot_s, self.hv_condop_s, \
                    self.hv_val1_type_s, self.hv_val1_like_s, self.hv_val1_bool_s, self.hv_val1_sp_s, self.hv_val1_ep_s, \
                    self.hv_val2_type_s, self.hv_val2_like_s, self.hv_val2_bool_s, self.hv_val2_sp_s, self.hv_val2_ep_s, \
                    self.hv_vu_col1_s, self.hv_vu_col2_s, self.hv_vu_agg1_s, self.hv_vu_agg2_s, self.hv_vu_dist1_s, self.hv_vu_dist2_s, self.hv_vu_op_s = generate_multiple_cu( qtok_summary, att_mask, q_tok_embed, col_embed, self.bert_q_mask, self.bert_col_mask, 2, "HAVING_CU", regularizer )

                    # Loss calculation.
                    prop_having_loss = 0.0
                    base_mask       = tf.logical_and( tf.sequence_mask( self.c_having_num, 2 ), tf.tile( tf.expand_dims( tf.cast( self.c_has_having, tf.bool ), -1 ), [ 1, 2 ] ) )
                    val1_se_mask    = tf.logical_and( base_mask, tf.equal( self.c_hv_val1_type, 0 ) )
                    val1_se_mask    = tf.logical_and( val1_se_mask, tf.logical_not( tf.cast( self.meta_hv_igval1, tf.bool ) ) )

                    val1_b_mask     = tf.logical_and( base_mask, tf.equal( self.c_hv_val1_type, 1 ) )
                    val1_b_mask     = tf.logical_and( val1_b_mask, tf.logical_not( tf.cast( self.meta_hv_igval1, tf.bool ) ) )

                    val2_mask       = tf.logical_and( base_mask, tf.equal( self.c_hv_condop, 0 ) )      # ONLY between case.
                    val2_se_mask    = tf.logical_and( val2_mask, tf.equal( self.c_hv_val2_type, 0 ) )
                    val2_se_mask    = tf.logical_and( val2_se_mask, tf.logical_not( tf.cast( self.meta_hv_igval2, tf.bool ) ) )

                    val2_b_mask     = tf.logical_and( val2_mask, tf.equal( self.c_hv_val2_type, 1 ) )
                    val2_b_mask     = tf.logical_and( val2_b_mask, tf.logical_not( tf.cast( self.meta_hv_igval2, tf.bool ) ) )

                    vu_val2_mask    = tf.logical_and( base_mask, tf.not_equal( self.c_hv_vu_op, 0 ) )
                    prop_having_loss += self._get_conditioned_loss( self.c_has_having, self.having_num_score, self.c_having_num )
                    prop_having_loss += self._get_conditioned_loss( base_mask, self.hv_agg_s, self.c_hv_agg )
                    prop_having_loss += self._get_conditioned_loss( base_mask, self.hv_isnot_s, self.c_hv_isnot )
                    prop_having_loss += self._get_conditioned_loss( base_mask, self.hv_condop_s, self.c_hv_condop )
                    prop_having_loss += self._get_conditioned_loss( base_mask, self.hv_val1_type_s, self.c_hv_val1_type )
                    prop_having_loss += self._get_conditioned_loss( val1_se_mask, self.hv_val1_like_s, self.c_hv_val1_like ) # Only when the text match is text span.
                    prop_having_loss += self._get_conditioned_loss( val1_b_mask, self.hv_val1_bool_s, self.c_hv_val1_bool )
                    prop_having_loss += self._get_conditioned_loss( val1_se_mask, self.hv_val1_sp_s, self.c_hv_val1_sp )
                    prop_having_loss += self._get_conditioned_loss( val1_se_mask, self.hv_val1_ep_s, self.c_hv_val1_ep )
                    prop_having_loss += self._get_conditioned_loss( val2_mask, self.hv_val2_type_s, self.c_hv_val2_type )
                    prop_having_loss += self._get_conditioned_loss( val2_se_mask, self.hv_val2_like_s, self.c_hv_val2_like )
                    prop_having_loss += self._get_conditioned_loss( val2_b_mask, self.hv_val2_bool_s, self.c_hv_val2_bool )
                    prop_having_loss += self._get_conditioned_loss( val2_se_mask, self.hv_val2_sp_s, self.c_hv_val2_sp )
                    prop_having_loss += self._get_conditioned_loss( val2_se_mask, self.hv_val2_ep_s, self.c_hv_val2_ep )
                    prop_having_loss += self._get_conditioned_loss( base_mask, self.hv_vu_op_s, self.c_hv_vu_op )
                    prop_having_loss += self._get_conditioned_loss( base_mask, self.hv_vu_agg1_s, self.c_hv_vu_agg1 )
                    prop_having_loss += self._get_conditioned_loss( base_mask, self.hv_vu_col1_s, self.c_hv_vu_col1 )
                    prop_having_loss += self._get_conditioned_loss( base_mask, self.hv_vu_dist1_s, self.c_hv_vu_dist1 )
                    prop_having_loss += self._get_conditioned_loss( vu_val2_mask, self.hv_vu_agg2_s, self.c_hv_vu_agg2 )
                    prop_having_loss += self._get_conditioned_loss( vu_val2_mask, self.hv_vu_col2_s, self.c_hv_vu_col2 )
                    prop_having_loss += self._get_conditioned_loss( vu_val2_mask, self.hv_vu_dist2_s, self.c_hv_vu_dist2 )

                # 12. Training & Others.
                self.tbl_cross_entropy  = prop_table_loss
                self.cross_entropy      = prop_class_loss + prop_gb_loss + prop_ob_loss + prop_limit_loss + prop_sel_loss + prop_where_loss + prop_having_loss

                reg_vars    = tf.get_collection( tf.GraphKeys.REGULARIZATION_LOSSES )
                l2_loss     = tf.contrib.layers.apply_regularization( regularizer, reg_vars )
                step_size       = 3
                self.learning_rate  = tf.cond( self.global_step < batch_num * step_size, \
                                                lambda: initial_lf, \
                                                lambda: tf.train.exponential_decay( initial_lf, self.global_step - batch_num * step_size, batch_num * step_size, 0.8, staircase = True ) )

                optimizer       = tf.train.AdamOptimizer( learning_rate = self.learning_rate )
                train_ops       = optimizer.minimize( self.cross_entropy + l2_loss )
                train_tbl_ops   = optimizer.minimize( self.tbl_cross_entropy + l2_loss )
                update_ops      = tf.get_collection( tf.GraphKeys.UPDATE_OPS )
                self.train_op   = tf.group( [ train_ops, update_ops ] )
                self.train_tbl_op   = tf.group( [ train_tbl_ops ] )

                if init_bert != None:
                    tvars           = tf.trainable_variables()
                    initialized_variable_names  = {}
                    ( assignment_map, initialized_variable_names )  = modeling.get_assignment_map_from_checkpoint( tvars, init_bert )
                    tf.train.init_from_checkpoint( init_bert, assignment_map )

                self.init       = tf.global_variables_initializer()
                self.saver      = tf.train.Saver()


