import os
import tensorflow as tf
import numpy as np
import pickle
import math
import random
import ast
import sys
sys.path.append( "./util" )
sys.path.append( "./util/bert" )

import modeling
import tokenization

import corpus_util
from data_storage       import DataStorage
from db_meta            import *
from sqlgen             import SQLGen
from testinfo_obj       import TestInfo
from timemeasurer       import TimeMeasurer
from progbar            import Progbar
from prop_train_manager import *
from actual_test        import *
from tok_id_ret         import *

class SQLTrainer:
    def __init__( self, bert_tokenizer = None, bert_config = None ):
        self.ds = DataStorage()
        self.st = SQLTester( bert_tokenizer = bert_tokenizer, bert_config = bert_config )
        self._bert_tokenizer    = bert_tokenizer
        self._bert_config       = bert_config

    def readCorpus( self, db_path, vec_train_path, dev_path ):
        self.ds.load_db_from_file( db_path )
        self.ds.load_train_datasets( vec_train_path, dev_path )

        # 1. Generate property-specific train / test data.
        self.vec_prop_train, self.vec_prop_valid    = self.ds.get_prop_cla_features( merged_col_name = True )
        self.vec_real_valid                         = self.ds.get_actual_input_features( merged_col_name = True )

        get_bert_tokens( self.vec_prop_train + self.vec_prop_valid,  self._bert_tokenizer )
        train_bert_removed  = 0
        valid_bert_removed  = 0
        final_train = []
        final_valid = []
        for v_f in self.vec_prop_train:
            bert_len    = 0
            bert_len    += 2 + len( v_f[ Q_BERT_TOK ] ) + 1 + len( v_f[ PF_PATH ] )
            for vec_tbl_bert in v_f[ C_BERT_TOK ]:
                for vec_col_bert in vec_tbl_bert:
                    bert_len    += 1 + len( vec_col_bert )
            bert_len    += 2    # *
            if bert_len >= 512:
                train_bert_removed  += 1
            else:
                final_train.append( v_f )
        for v_f in self.vec_prop_valid:
            bert_len    = 0
            bert_len    += 2 + len( v_f[ Q_BERT_TOK ] ) + 1 + len( v_f[ PF_PATH ] )
            for vec_tbl_bert in v_f[ C_BERT_TOK ]:
                for vec_col_bert in vec_tbl_bert:
                    bert_len    += 1 + len( vec_col_bert )
            bert_len    += 2    # *
            if bert_len >= 512:
                valid_bert_removed  += 1
            else:
                final_valid.append( v_f )
        self.vec_prop_train = final_train
        self.vec_prop_valid = final_valid
        print ( "TOO LARGE FOR BERT (TRAIN): ", train_bert_removed )
        print ( "TOO LARGE FOR BERT (VALID): ", valid_bert_removed )

        get_bert_tokens( self.vec_real_valid, self._bert_tokenizer )
            
        self.sg = SQLGen( bert_config = self._bert_config )

        # Extract Path IDX.
        for v_f in self.vec_prop_train + self.vec_prop_valid:
            v_f[ PF_PATHIDX ] = [ IDX_PATH[ str_path ] for str_path in v_f[ PF_PATH ] ]

        # Extract Actually used tables.
        # Remove those required to JOIN, reducing the garbage inputs.
        train_reduce_num    = 0
        valid_reduce_num    = 0
        vec_filter_codes    = [ [ OF_VU_COL1, OF_VU_OPERATOR, OF_VU_COL2 ], [ SF_VU_COL1, SF_VU_OPERATOR, SF_VU_COL2 ], \
                            [ WF_CU_VU_COL1, WF_CU_VU_OPERATOR, WF_CU_VU_COL2 ], [ HV_CU_VU_COL1, HV_CU_VU_OPERATOR, WF_CU_VU_COL2 ] ]

        for v_f in self.vec_prop_train + self.vec_prop_valid:
            if 0 not in v_f[ SF_VU_COL1 ]:  # Try to reduce. ( NO STAR )
                set_used_col_idx    = set()
                for c1_code, op_code, c2_code in vec_filter_codes:
                    for col1, op, col2 in zip( v_f[ c1_code ], v_f[ op_code ], v_f[ c2_code ] ):
                        set_used_col_idx.add( col1 )
                        if op > 0:
                            set_used_col_idx.add( col2 )
                
                db  = v_f[ META_DB ]
                set_used_tbl_idx    = set()
                for cidx in set_used_col_idx:
                    if db.vec_cols[ cidx ].table_belong != None:
                        set_used_tbl_idx.add( db.vec_cols[ cidx ].table_belong.tbl_idx )      

                if set( v_f[ TV_TABLES_USED_IDX ] ) != set_used_tbl_idx:
                    recover = self._generate_actual_tbls( db, list( set_used_tbl_idx ) )
                    if set( recover ) == set( v_f[ TV_TABLES_USED_IDX ] ):
                        if v_f in self.vec_prop_train:
                            train_reduce_num    += 1
                        else:
                            valid_reduce_num    += 1

                        v_f[ TV_TABLES_USED_IDX ]   = list( set_used_tbl_idx )
                        v_f[ TV_TABLES_NUM ]        = len( v_f[ TV_TABLES_USED_IDX ] )

        print ( "TRAIN REDUCE NUM: [%d]" % train_reduce_num )
        print ( "VALID REDUCE NUM: [%d]" % valid_reduce_num )

        print ( "PROPERTY TRAIN INSTANCE NUM: [%d]" % len( self.vec_prop_train ) )
        print ( "PROPERTY VALID INSTANCE NUM: [%d]" % len( self.vec_prop_valid ) )
        
    # Test on the features.
    def _do_test( self, sess, target_data, batch_size ):
        batch_num   = int( len( target_data ) / batch_size )
        if len( target_data ) % batch_size != 0:
            batch_num   += 1

        ti  = TestInfo()
        add_prop_test_info( ti, self.sg )
        add_tbl_test_info( ti, self.sg )

        prog        = Progbar( target = batch_num )
        for batch_idx in range( batch_num ):
            vec_data                        = target_data[ batch_idx * batch_size: min( ( batch_idx + 1 ) * batch_size, len( target_data ) ) ]

            # 1. Classify Tbl.
            fdv_tbl                         = prepare_tbl_dict( vec_data, bert_tokenizer = self._bert_tokenizer )
            fdv_tbl[ self.sg.is_train ]     = False
            fdv_tbl[ self.sg.drop_rate ]    = 0.0
            tbl_results                     = ti.fetch_tbl_tensor_info( sess, fdv_tbl )

            # 2. Get the extracted tbl info.
            vec_tbl_idx_score   = tbl_results[ TV_TABLES_USED_IDX ]
            vec_tbl_num         = tbl_results[ TV_TABLES_NUM ] 
        
            vec_mod_tbl_num     = []
            vec_tbl_extracted   = []
            for tbl_idx_score, tbl_num in zip( vec_tbl_idx_score, vec_tbl_num ):
                vec_tbl_idx     = np.array( tbl_idx_score ).argsort()[ -tbl_num: ]
                vec_tbl_idx     = sorted( vec_tbl_idx.tolist() )
                vec_tbl_extracted.append( vec_tbl_idx )

            # 3. Classify Props.
            fdv_prop                        = prepare_prop_dict( vec_data, vec_tbl_extracted, bert_tokenizer = self._bert_tokenizer )
            fdv_prop[ self.sg.is_train ]    = False
            fdv_prop[ self.sg.drop_rate ]   = 0.0
            prop_results                    = ti.fetch_prop_tensor_info( sess, fdv_prop )

            # 4. Update Pointer results to the actual columns.
            vec_score_answers   = [ [ GF_NUMCOL, [ GF_COLLIST ] ], \
                                    [ OF_NUMVU, [ OF_VU_COL1, OF_VU_COL2, OF_VU_AGG1, OF_VU_AGG2, OF_VU_DIST1, OF_VU_DIST2, OF_VU_OPERATOR ] ], \
                                    [ SF_NUM_VU, [ SF_VU_AGGALL, SF_VU_COL1, SF_VU_COL2, SF_VU_AGG1, SF_VU_AGG2, SF_VU_DIST1, SF_VU_DIST2, SF_VU_OPERATOR ] ], \
                                    [ WF_NUM_CONDUNIT, [ WF_CU_AGGREGATOR, WF_CU_IS_NOT, WF_CU_COND_OP, \
                                                        WF_CU_VAL1_TYPE, WF_CU_VAL1_SP, WF_CU_VAL1_EP, WF_CU_VAL1_LIKELY, WF_CU_VAL1_BOOLVAL, \
                                                        WF_CU_VAL2_TYPE, WF_CU_VAL2_SP, WF_CU_VAL2_EP, WF_CU_VAL2_LIKELY, WF_CU_VAL2_BOOLVAL, \
                                                        WF_CU_VU_OPERATOR, WF_CU_VU_AGG1, WF_CU_VU_COL1, WF_CU_VU_DIST1, WF_CU_VU_AGG2, WF_CU_VU_COL2, WF_CU_VU_DIST2 ] ], \
                                    [ HV_NUM_CONDUNIT, [ HV_CU_AGGREGATOR, HV_CU_IS_NOT, HV_CU_COND_OP, \
                                                         HV_CU_VAL1_TYPE, HV_CU_VAL1_SP, HV_CU_VAL1_EP, HV_CU_VAL1_LIKELY, HV_CU_VAL1_BOOLVAL, \
                                                         HV_CU_VAL2_TYPE, HV_CU_VAL2_SP, HV_CU_VAL2_EP, HV_CU_VAL2_LIKELY, HV_CU_VAL2_BOOLVAL, \
                                                         HV_CU_VU_OPERATOR, HV_CU_VU_AGG1, HV_CU_VU_COL1, HV_CU_VU_DIST1, HV_CU_VU_AGG2, HV_CU_VU_COL2, HV_CU_VU_DIST2 ] ] ]

            for num_col_name, vec_col_name in vec_score_answers:
                for col_name in vec_col_name:
                    prop_results[ col_name ]    = [ np.argmax( v, -1 ).tolist()[ :l ] for v, l in zip( prop_results[ col_name ], prop_results[ num_col_name ] ) ]

            # 5. Update the extracted column indexes
            vec_col_pointers    = [ GF_COLLIST, OF_VU_COL1, OF_VU_COL2, SF_VU_COL1, SF_VU_COL2, WF_CU_VU_COL1, WF_CU_VU_COL2, HV_CU_VU_COL1, HV_CU_VU_COL2 ]
            for pidx, prop_data in enumerate( vec_data ):
                db  = prop_data[ META_DB ]

                # 1. Get the valid column span. ( Inclusive )
                vec_col_spans   = [ [ 0, 0 ] ]  # Special colum: *
                for t in vec_tbl_extracted[ pidx ]:
                    if t < len( db.vec_tbls ):
                        tbl = db.vec_tbls[ t ]
                        vec_col_spans.append( [ tbl.vec_cols[0].col_idx, tbl.vec_cols[-1].col_idx ] )
                    else:
                        print ( "ERROR: %d VS. %d" % ( t, len( db.vec_tbls ) ) )
                vec_col_spans   = sorted( vec_col_spans, key = lambda x: x[0] )

                # 2. Generate New Col Idx - Old Col Idx Map.
                new_idx     = 0
                map_col_idx = dict()
                for sidx, eidx in vec_col_spans:
                    for col_idx in range( sidx, eidx + 1 ):
                        map_col_idx[ new_idx ]  = col_idx
                        new_idx += 1
                
                # 3. Update the mappings.
                for col_name in vec_col_pointers:
                    prop_results[ col_name ][ pidx ]    = [ map_col_idx[ col_idx ] for col_idx in prop_results[ col_name ][ pidx ] if col_idx in map_col_idx]
            
            # 6. Get merged results & evaluate.
            total_results   = prop_results
            for k, v in tbl_results.items():
                total_results[k]    = v
            
            ti.integrate_eval_result( total_results, vec_data )
            prog.update( batch_idx + 1 )

        return ti

    def train( self, BERT_DIR, lf, save_path, batch_size ):
        batch_num   = int( len( self.vec_prop_train ) / batch_size )
        if len( self.vec_prop_train ) % batch_size != 0:
            batch_num   += 1

        self.sg.constructGraph( batch_num, lf, init_bert = os.path.join( BERT_DIR, "bert_model.ckpt" ) )

        config                          = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement     = True
        with tf.Session( graph = self.sg.graph, config = config ) as sess:
            ckpt = tf.train.get_checkpoint_state( save_path )
            v2_path = ckpt.model_checkpoint_path + ".index" if ckpt else ""
            self.sg.init.run()

            tvars           = tf.trainable_variables()
            initialized_variable_names  = {}
            ( assignment_map, initialized_variable_names )  = modeling.get_assignment_map_from_checkpoint( tvars, os.path.join( BERT_DIR, "bert_model.ckpt" ) )

            tf.logging.info("**** Trainable Variables ****")
            for var in tvars:
                init_string = ""
                if var.name in initialized_variable_names:
                    init_string = ", *INIT_FROM_CKPT*"
                print("  name = %s, shape = %s%s" % ( var.name, var.shape, init_string ) )

            print ( 'Num params: %d' % sum(v.get_shape().num_elements() for v in tf.trainable_variables()) )
        
            step    = 1
            valid_accept_cnt    = 0
            global_step         = 0
            prev_valid_acc      = 0.0
            while True:
                self.vec_prop_train = corpus_util.shuffleData( self.vec_prop_train )       # JUST FOR DEBUGGING
                print ( "STEP: %d" % step )
                average_train_loss  = 0.0
                average_valid_loss  = 0.0
                
                tm  = TimeMeasurer()
                tm.start( "train" )

                prog    = Progbar( target = batch_num )
                for batch_idx in range( batch_num ):
                    global_step += 1
                    vec_prop_data   = self.vec_prop_train[ batch_idx * batch_size: min( ( batch_idx + 1 ) * batch_size, len( self.vec_prop_train ) ) ]
            
                    # 1. Train the Table Selector Network.
                    feed_dict_train                         = prepare_tbl_dict( vec_prop_data, bert_tokenizer = self._bert_tokenizer )
                    feed_dict_train[ self.sg.global_step ]  = global_step
                    feed_dict_train[ self.sg.is_train ]     = True
                    feed_dict_train[ self.sg.drop_rate ]    = 0.1
            
                    _, loss_val = sess.run( [ self.sg.train_tbl_op, self.sg.tbl_cross_entropy ], feed_dict = feed_dict_train )
                    average_train_loss  += loss_val * len( vec_prop_data )

                    if math.isnan( loss_val ) or math.isinf( loss_val ):
                        print ( "TBL NAN/INF ERROR" )
                        sys.exit( 0 )

                    # 2. Train for Other property selection network.
                    #    Feed the used tables as its inputs.
                    vec_valid_tbl_idx                       = [ v_f[ TV_TABLES_USED_IDX ] for v_f in vec_prop_data ]
                    feed_dict_prop_train                    = prepare_prop_dict( vec_prop_data, vec_valid_tbl_idx, bert_tokenizer = self._bert_tokenizer  )
                    feed_dict_prop_train[ self.sg.global_step ]  = global_step
                    feed_dict_prop_train[ self.sg.is_train ]     = True
                    feed_dict_prop_train[ self.sg.drop_rate ]    = 0.1
                    
                    _, loss_val = sess.run( [ self.sg.train_op, self.sg.cross_entropy ], feed_dict = feed_dict_prop_train )
                    average_train_loss  += loss_val * len( vec_prop_data )
                    if math.isnan( loss_val ) or math.isinf( loss_val ):
                        print ( "PROP NAN/INF ERROR" )
                        sys.exit( 0 )

                    prog.update( batch_idx + 1, [ ( "train loss", loss_val ) ] )

                average_train_loss /= len( self.vec_prop_train )
                tm.end( "train" )

                print ( "Average Training Loss: " + str( average_train_loss ) )
                print ( "Elapsed Per Step: " + str( tm.getElapsed( "train" ) ) )

                ti  = self._do_test( sess, self.vec_prop_valid, batch_size )
                ti.print_eval_results()
                em_f1   = ti.get_overall_result()

                valid_acc   = em_f1
                if valid_acc < prev_valid_acc:
                    if valid_accept_cnt >= 20:
                        print ( "Training Finished." )
                        return
                    else:
                        valid_accept_cnt    += 1
                        print ( "TOLERATING CNT: %d" % valid_accept_cnt )
                else:
                    self._save_model( sess, save_path )
                    print ( "MODEL SAVED." )
                    prev_valid_acc      = valid_acc
                    valid_accept_cnt    = 0

                step    += 1

    def test( self, load_path, batch_size ):
        self.sg.constructGraph()

        config                          = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement     = True
        with tf.Session( graph = self.sg.graph, config = config ) as sess:
            self.sg.init.run()

            self._load_model( sess, load_path )

            tm  = TimeMeasurer()
            tm.start( "test" )
            ti   = self._do_test( sess, self.vec_prop_valid, batch_size )
            tm.end( "test" )
            elapsed = tm.getElapsed( "test" )
            print ( "elased_inmsec = " + str( elapsed ) + "( per sen: " + str( float( elapsed ) / len( self.vec_prop_valid ) ) + " )" )
            ti.print_eval_results()
                

    def _save_model( self, sess, save_path ):
        self.sg.saver.save( sess, "%s/best_chk" % save_path )

    def _load_model( self, sess, save_path ):
        ckpt    = tf.train.get_checkpoint_state( save_path )
        v2_path = ckpt.model_checkpoint_path + ".index" if ckpt else ""

        self.sg.saver.restore( sess, ckpt.model_checkpoint_path )

    def _generate_actual_tbls( self, db, vec_tbl_idx ):
        if len( vec_tbl_idx ) == 1:
            return vec_tbl_idx

        # 1. Find the "Join Path".
        joinable_tbls   = dict()
        for f1, f2 in db.foreign_keys:
            t1  = f1.table_belong.tbl_idx
            t2  = f2.table_belong.tbl_idx
            if t1 not in joinable_tbls:
                joinable_tbls[ t1 ] = set()
            if t2 not in joinable_tbls:
                joinable_tbls[ t2 ] = set()
            joinable_tbls[ t1 ].add( t2 )
            joinable_tbls[ t2 ].add( t1 )

        cur_path_cand   = [ set( [ v ] ) for v in vec_tbl_idx ]
        final_path      = None
        while len( cur_path_cand ) > 0:
            new_path_cand   = []
            for path_cand in cur_path_cand:
                for t in path_cand:
                    if t not in joinable_tbls:
                        continue
                    for new_exp in joinable_tbls[ t ]:
                        if new_exp not in path_cand:
                            new_path            = path_cand.copy()
                            new_path.add( new_exp )

                            if set( vec_tbl_idx ) <= new_path:
                                final_path  = new_path
                                break
                            if new_path not in new_path_cand:
                                new_path_cand.append( set( new_path ) )

                    if final_path != None:
                        break
                if final_path != None:
                    break
            if final_path != None:
                break

            cur_path_cand   = new_path_cand

        if final_path == None:
            return vec_tbl_idx

        return final_path


if __name__== "__main__":
    if len( sys.argv ) < 3: 
        print (  "Usage: python trainer.py [BERT_MODEL_DIR] [DATA_DIR]" )
    BERT_DIR        = sys.argv[1]
    DATA_DIR        = sys.argv[2]
    bert_tokenizer  = tokenization.FullTokenizer( vocab_file = os.path.join( BERT_DIR, "vocab.txt" ), do_lower_case = True )
    bert_config     = modeling.BertConfig.from_json_file( os.path.join( BERT_DIR, "bert_config.json" ) )
    st  = SQLTrainer( bert_tokenizer = bert_tokenizer, bert_config = bert_config )
    st.readCorpus( os.path.join( DATA_DIR, "tables.json" ), [ os.path.join( DATA_DIR, "train_spider.json" ), os.path.join( DATA_DIR, "train_others.json" ) ], os.path.join( DATA_DIR, "dev.json" ) )

    st.train( BERT_DIR, lf = 1e-5, save_path = "checkpoint", batch_size = 4 )
    st.test( load_path = "checkpoint", batch_size = 4 )

