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

from data_storage       import DataStorage
from db_meta            import *
from sqlgen             import SQLGen
from testinfo_obj       import TestInfo
from timemeasurer       import TimeMeasurer
from progbar            import Progbar
from prop_train_manager import *
from tok_id_ret         import *

def copy_dict( source ):
    ret = dict()
    for k, v in source.items():
        ret[k]  = v
    return ret

from multiprocessing.pool import ThreadPool
def threading_func( tester, sess, sg, vec_data, ti ):
    if True:
        if True:
            ev_elist        = []
            vec_result_q    = []
            vec_exe_q       = []
                
            # 1. Assign temporal QIDs & PATH,
            for qidx, data in enumerate( vec_data ):
                data[ "QID" ]   = qidx
                data[ PF_PATH ] = [ PATH_NONE ]
                data[ PF_PATHIDX ] = [ IDX_PATH[ str_path ] for str_path in data[ PF_PATH ] ]
                vec_exe_q.append( data )

            # 2. Run the batch based on the execution Q info.
            exe_iter_num = 0
            while len( vec_exe_q ) > 0 and exe_iter_num < 4 :
                exe_iter_num += 1
                # 2-1. Classify Tbl.
                fdv_tbl                    = prepare_tbl_dict( vec_exe_q, inference = True, bert_tokenizer = tester._bert_tokenizer )
                fdv_tbl[ sg.is_train ]     = False
                fdv_tbl[ sg.drop_rate ]    = 0.0
                tbl_results                = ti.fetch_tbl_tensor_info( sess, fdv_tbl )

                # 2-2. Get the extracted tbl info.
                vec_tbl_idx_score   = tbl_results[ TV_TABLES_USED_IDX ]
                vec_tbl_num         = tbl_results[ TV_TABLES_NUM ]

                vec_tbl_extracted   = []
                for tbl_idx_score, tbl_num in zip( vec_tbl_idx_score, vec_tbl_num ):
                    vec_tbl_idx = np.array( tbl_idx_score ).argsort()[ -tbl_num: ]
                    vec_tbl_idx = sorted( vec_tbl_idx.tolist() )
                    vec_tbl_extracted.append( vec_tbl_idx )
                
                # 2-3. Classify Props.
                fdv_prop                    = prepare_prop_dict( vec_exe_q, vec_tbl_extracted, inference = True, bert_tokenizer = tester._bert_tokenizer )
                fdv_prop[ sg.is_train ]     = False
                fdv_prop[ sg.drop_rate ]    = 0.0
                prop_results                = ti.fetch_prop_tensor_info( sess, fdv_prop )

                # 2-4. Update Pointer results to the actual columns.
                vec_score_answers   = [ [ WF_NUM_CONDUNIT, [ WF_CU_AGGREGATOR, WF_CU_IS_NOT, WF_CU_COND_OP, \
                                                            WF_CU_VAL1_TYPE, WF_CU_VAL1_SP, WF_CU_VAL1_EP, WF_CU_VAL1_LIKELY, WF_CU_VAL1_BOOLVAL, \
                                                            WF_CU_VAL2_TYPE, WF_CU_VAL2_SP, WF_CU_VAL2_EP, WF_CU_VAL2_LIKELY, WF_CU_VAL2_BOOLVAL, \
                                                            WF_CU_VU_OPERATOR, WF_CU_VU_AGG1, WF_CU_VU_COL1, WF_CU_VU_DIST1, WF_CU_VU_AGG2, WF_CU_VU_COL2, WF_CU_VU_DIST2 ] ], \
                                        [ HV_NUM_CONDUNIT, [ HV_CU_AGGREGATOR, HV_CU_IS_NOT, HV_CU_COND_OP, \
                                                             HV_CU_VAL1_TYPE, HV_CU_VAL1_SP, HV_CU_VAL1_EP, HV_CU_VAL1_LIKELY, HV_CU_VAL1_BOOLVAL, \
                                                             HV_CU_VAL2_TYPE, HV_CU_VAL2_SP, HV_CU_VAL2_EP, HV_CU_VAL2_LIKELY, HV_CU_VAL2_BOOLVAL, \
                                                             HV_CU_VU_OPERATOR, HV_CU_VU_AGG1, HV_CU_VU_COL1, HV_CU_VU_DIST1, HV_CU_VU_AGG2, HV_CU_VU_COL2, HV_CU_VU_DIST2 ] ] ]
                prop_results    = tester._selection_dup_remove( prop_results )
                prop_results    = tester._orderby_dup_remove( prop_results )

                for num_col_name, vec_col_name in vec_score_answers:
                    for col_name in vec_col_name:
                        prop_results[ col_name ]    = [ np.argsort( v, -1 ).tolist()[ :l ] for v, l in zip( prop_results[ col_name ], prop_results[ num_col_name ] ) ]  # BS X L (# of units) X C (# of candidates)
                        prop_results[ col_name ]    = [ [ unit[::-1] for unit in batch ] for batch in prop_results[ col_name ] ]
            
                prop_results[ GF_COLLIST ]  = [ np.argmax(  v, -1 ).tolist()[ :l ] for v, l in zip( prop_results[ GF_COLLIST ], prop_results[ GF_NUMCOL ] ) ]

                # 2-5. Update the extracted column indexes.
                vec_col_pointers    = [ GF_COLLIST, OF_VU_COL1, OF_VU_COL2, SF_VU_COL1, SF_VU_COL2, WF_CU_VU_COL1, WF_CU_VU_COL2, HV_CU_VU_COL1, HV_CU_VU_COL2 ]
                set_col_vecs        = set( [ WF_CU_VU_COL1, WF_CU_VU_COL2, HV_CU_VU_COL1, HV_CU_VU_COL2 ] )
                for pidx, prop_data in enumerate( vec_exe_q ):
                    db  = prop_data[ META_DB ]

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
                        if col_name in set_col_vecs:
                            prop_results[ col_name ][ pidx ]    = [ [ map_col_idx[ col_idx ] for col_idx in vec_col_idx if col_idx in map_col_idx ] for vec_col_idx in prop_results[ col_name ][ pidx ] ]
                        else:
                            prop_results[ col_name ][ pidx ]    = [ map_col_idx[ col_idx ] for col_idx in prop_results[ col_name ][ pidx ] if col_idx in map_col_idx ]
            
                # 2-6. Store merged results.
                total_results   = prop_results
                total_results[ TV_TABLES_USED_IDX ] = vec_tbl_extracted

                vec_new_exe_q   = []
                for exe_idx, exe_info in enumerate( vec_exe_q ):
                    exe_result  = copy_dict( exe_info )
                    for k, v in total_results.items():
                        exe_result[k]   = v[ exe_idx ]
                    vec_result_q.append( exe_result )

                    # Generate new Exe queue, if required.
                    if exe_result[ PF_MERGEOP ] > 0 and exe_result[ PF_PATH ] == [ PATH_NONE ]:
                        path_type   = PATH_UNION
                        if exe_result[ PF_MERGEOP ] == 2:
                            path_type   = PATH_INTER
                        elif exe_result[ PF_MERGEOP ] == 3:
                            path_type   = PATH_EXCEPT

                        new_exe = copy_dict( exe_info )
                        new_exe[ PF_PATH ]  = [ path_type ]
                        new_exe[ PF_PATHIDX ] = [ IDX_PATH[ str_path ] for str_path in new_exe[ PF_PATH ] ]
                        vec_new_exe_q.append( new_exe )
                    if exe_result[ PF_WHERE ]:
                        sel_cond_idx    = 0
                        cur_w_path  = exe_info[ PF_PATH ].copy()
                        for cond_idx in range( exe_result[ WF_NUM_CONDUNIT ] ):
                            if exe_result[ WF_CU_VAL1_TYPE ][ cond_idx ][0] == 2:  # Select Statement.
                                new_exe = copy_dict( exe_info )
                                new_exe[ PF_PATH ]  = cur_w_path.copy()
                                if new_exe[ PF_PATH ] == [ PATH_NONE ]:
                                    new_exe[ PF_PATH ]  = []
                                if sel_cond_idx > 0:
                                    new_exe[ PF_PATH ].append( PATH_PAR )
                                new_exe[ PF_PATH ].append( PATH_WHERE )
                                cur_w_path      = new_exe[ PF_PATH ].copy()
                                new_exe[ PF_PATHIDX ] = [ IDX_PATH[ str_path ] for str_path in new_exe[ PF_PATH ] ]
                                sel_cond_idx    += 1
                                vec_new_exe_q.append( new_exe )
                    if exe_result[ PF_HAVING ]:
                        sel_cond_idx    = 0
                        cur_h_path  = exe_info[ PF_PATH ].copy()
                        for cond_idx in range( exe_result[ HV_NUM_CONDUNIT ] ):
                            if exe_result[ HV_CU_VAL1_TYPE ][ cond_idx ][0] == 2:  # Select Statement.
                                new_exe = copy_dict( exe_info )
                                new_exe[ PF_PATH ]  = cur_h_path.copy()
                                if new_exe[ PF_PATH ] == [ PATH_NONE ]:
                                    new_exe[ PF_PATH ]  = []
                                if sel_cond_idx > 0:
                                    new_exe[ PF_PATH ].append( PATH_PAR )
                                new_exe[ PF_PATH ].append( PATH_HAVING )
                                cur_h_path  = new_exe[ PF_PATH ].copy()
                                new_exe[ PF_PATHIDX ] = [ IDX_PATH[ str_path ] for str_path in new_exe[ PF_PATH ] ]
                                sel_cond_idx    += 1
                                vec_new_exe_q.append( new_exe )
                vec_exe_q   = vec_new_exe_q

            # 3. Use the result Q to get the final statement.
            for qidx, data in enumerate( vec_data ):
                vec_result_for_q    = []
                for result in vec_result_q:
                    if result[ "QID" ] == qidx:
                        vec_result_for_q.append( result )

                sql_statement   = tester.generate_statement( vec_result_for_q )
                ev_elist.append( [ sql_statement, data[ META_DB ].db_id ] )
            return '\n'.join( [ "%s\t%s" % ( v[0], v[1] ) for v in ev_elist ] )


class SQLTester:
    def __init__( self, f_gold, f_db, f_table, bert_tokenizer = None, bert_config = None ):
        self._bert_tokenizer    = bert_tokenizer
        self._bert_config       = bert_config

        # evaluate info.
        self._ev_kmaps  = build_foreign_key_map_from_json( f_table )
        with open( f_gold ) as f:
            self._ev_glist  = [ l.strip().split('\t') for l in f.readlines() if len( l.strip() ) > 0 ]
        self._ev_dbdir  = f_db

    def readCorpus( self, db_path, dev_path ):
        ds = DataStorage()
        ds.load_db_from_file( db_path )
        ds.load_valid_datasets( dev_path )

        self.vec_prop_valid = ds.get_actual_input_features( merged_col_name = True )
        get_bert_tokens( self.vec_prop_valid,  self._bert_tokenizer )

    # prop_results: Batch of the result.
    # Goal: Turn select-related elements into select column ID.
    #       If Duplicated, Choose the next priority one.
    def _selection_dup_remove( self, prop_results ):
        t_aggall    = []
        t_col1      = []
        t_col2      = []
        t_agg1      = []
        t_agg2      = []
        t_dist1     = []
        t_dist2     = []
        t_op        = []
        for nvu, aggall, col1, col2, agg1, agg2, dist1, dist2, op in zip( \
            prop_results[ SF_NUM_VU ], prop_results[ SF_VU_AGGALL ], \
            prop_results[ SF_VU_COL1 ], prop_results[ SF_VU_COL2 ], prop_results[ SF_VU_AGG1 ], prop_results[ SF_VU_AGG2 ], \
            prop_results[ SF_VU_DIST1 ], prop_results[ SF_VU_DIST2 ], prop_results[ SF_VU_OPERATOR ] ):

            set_shown   = set()

            v_aggall    = []
            v_col1      = []
            v_col2      = []
            v_agg1      = []
            v_agg2      = []
            v_dist1     = []
            v_dist2     = []
            v_op        = []
            for vu_idx in range( nvu ):
                s_aggall    = np.argsort( aggall[ vu_idx ] )[::-1].tolist()
                s_col1      = np.argsort( col1[ vu_idx ] )[::-1].tolist()
                s_col2      = np.argsort( col2[ vu_idx ] )[::-1].tolist()
                s_agg1      = np.argsort( agg1[ vu_idx ] )[::-1].tolist()
                s_agg2      = np.argsort( agg2[ vu_idx ] )[::-1].tolist()
                s_dist1     = np.argsort( dist1[ vu_idx ] )[::-1].tolist()
                s_dist2     = np.argsort( dist2[ vu_idx ] )[::-1].tolist()
                s_op        = np.argsort( op[ vu_idx ] )[::-1].tolist()
                vec_target  = [ s_aggall, s_col1, s_col2, s_agg1, s_agg2, s_dist1, s_dist2, s_op ]
                vec_indices = [ 0, 0, 0, 0, 0, 0, 0, 0 ]

                changed     = False
                while True:
                    key = []
                    if s_op[ vec_indices[7] ] == 0: # Only col1 is valid.
                        vec_dc_indices  = [ 0, 1, 3, 5 ]
                        for dc_idx in vec_dc_indices:
                            key.append( vec_target[ dc_idx ][ vec_indices[ dc_idx ] ] )

                    else:       # Both columns are valid.
                        for v, idx in zip( vec_target, vec_indices ):
                            key.append( v[ idx ] )
                    str_key = ' '.join( [ str(v) for v in key ] )
                    if str_key in set_shown:
                        if s_aggall[ vec_indices[0] ] == 0:             # Aggregator NONE: Change col 1.
                            vec_indices[1]  += 1
                            if vec_indices[1] >= len( s_col1 ):
                                break
                        else:                                           # Change Aggregator.
                            vec_indices[0]  += 1
                            if vec_indices[0] >= len( s_aggall ):
                                break
                        changed = True
                    else:
                        set_shown.add( str_key )
                        v_aggall.append( s_aggall[ vec_indices[0] ] )
                        v_col1.append( s_col1[ vec_indices[1] ] )
                        v_col2.append( s_col2[ vec_indices[2] ] )
                        v_agg1.append( s_agg1[ vec_indices[3] ] )
                        v_agg2.append( s_agg2[ vec_indices[4] ] )
                        v_dist1.append( s_dist1[ vec_indices[5] ] )
                        v_dist2.append( s_dist2[ vec_indices[6] ] )
                        v_op.append( s_op[ vec_indices[7] ] )
                        break
        
            t_aggall.append( v_aggall )
            t_col1.append( v_col1 )
            t_col2.append( v_col2 )
            t_agg1.append( v_agg1 )
            t_agg2.append( v_agg2 )
            t_dist1.append( v_dist1 )
            t_dist2.append( v_dist2 )
            t_op.append( v_op )
        prop_results[ SF_VU_AGGALL ]    = t_aggall
        prop_results[ SF_VU_COL1 ]      = t_col1
        prop_results[ SF_VU_COL2 ]      = t_col2
        prop_results[ SF_VU_AGG1 ]      = t_agg1
        prop_results[ SF_VU_AGG2 ]      = t_agg2
        prop_results[ SF_VU_DIST1 ]     = t_dist1
        prop_results[ SF_VU_DIST2 ]     = t_dist2
        prop_results[ SF_VU_OPERATOR ]  = t_op

        return prop_results
    # prop_results: Batch of the result.
    # Goal: Turn select-related elements into select column ID.
    #       If Duplicated, Choose the next priority one.
    def _orderby_dup_remove( self, prop_results ):
        t_col1      = []
        t_col2      = []
        t_agg1      = []
        t_agg2      = []
        t_dist1     = []
        t_dist2     = []
        t_op        = []
        for nvu, col1, col2, agg1, agg2, dist1, dist2, op in zip( \
            prop_results[ OF_NUMVU ], \
            prop_results[ OF_VU_COL1 ], prop_results[ OF_VU_COL2 ], prop_results[ OF_VU_AGG1 ], prop_results[ OF_VU_AGG2 ], \
            prop_results[ OF_VU_DIST1 ], prop_results[ OF_VU_DIST2 ], prop_results[ OF_VU_OPERATOR ] ):

            set_shown   = set()

            v_col1      = []
            v_col2      = []
            v_agg1      = []
            v_agg2      = []
            v_dist1     = []
            v_dist2     = []
            v_op        = []
            for vu_idx in range( nvu ):
                s_col1      = np.argsort( col1[ vu_idx ] )[::-1].tolist()
                s_col2      = np.argsort( col2[ vu_idx ] )[::-1].tolist()
                s_agg1      = np.argsort( agg1[ vu_idx ] )[::-1].tolist()
                s_agg2      = np.argsort( agg2[ vu_idx ] )[::-1].tolist()
                s_dist1     = np.argsort( dist1[ vu_idx ] )[::-1].tolist()
                s_dist2     = np.argsort( dist2[ vu_idx ] )[::-1].tolist()
                s_op        = np.argsort( op[ vu_idx ] )[::-1].tolist()
                vec_target  = [ s_col1, s_col2, s_agg1, s_agg2, s_dist1, s_dist2, s_op ]
                vec_indices = [ 0, 0, 0, 0, 0, 0, 0 ]

                changed     = False
                while True:
                    key = []
                    for v, idx in zip( vec_target, vec_indices ):
                        key.append( v[ idx ] )
                    str_key = ' '.join( [ str(v) for v in key ] )
                
                    if str_key in set_shown or ( s_agg1[ vec_indices[2] ] == 0 and s_col1[ vec_indices[0] ] == 0 ) :  # No * for ORDER-BY!
                        vec_indices[0]  += 1
                        if vec_indices[0] >= len( s_col1 ):   # Change col 1.
                            break
                        changed = True
                    else:
                        set_shown.add( str_key )
                        v_col1.append( s_col1[ vec_indices[0] ] )
                        v_col2.append( s_col2[ vec_indices[1] ] )
                        v_agg1.append( s_agg1[ vec_indices[2] ] )
                        v_agg2.append( s_agg2[ vec_indices[3] ] )
                        v_dist1.append( s_dist1[ vec_indices[4] ] )
                        v_dist2.append( s_dist2[ vec_indices[5] ] )
                        v_op.append( s_op[ vec_indices[6] ] )
                        break
        
            t_col1.append( v_col1 )
            t_col2.append( v_col2 )
            t_agg1.append( v_agg1 )
            t_agg2.append( v_agg2 )
            t_dist1.append( v_dist1 )
            t_dist2.append( v_dist2 )
            t_op.append( v_op )
        prop_results[ OF_VU_COL1 ]      = t_col1
        prop_results[ OF_VU_COL2 ]      = t_col2
        prop_results[ OF_VU_AGG1 ]      = t_agg1
        prop_results[ OF_VU_AGG2 ]      = t_agg2
        prop_results[ OF_VU_DIST1 ]     = t_dist1
        prop_results[ OF_VU_DIST2 ]     = t_dist2
        prop_results[ OF_VU_OPERATOR ]  = t_op

        return prop_results

    def _get_used_table( self, total_results ):
        vec_col_pointers    = [ [ PF_ORDERBY, OF_NUMVU, [ OF_VU_COL1, OF_VU_COL2 ] ], \
                                [ None, SF_NUM_VU, [ SF_VU_COL1, SF_VU_COL2 ] ], \
                                [ PF_WHERE, WF_NUM_CONDUNIT, [ WF_CU_VU_COL1, WF_CU_VU_COL2 ] ], \
                                [ PF_HAVING, HV_NUM_CONDUNIT, [ HV_CU_VU_COL1, HV_CU_VU_COL2 ] ] ]
        db                  = total_results[ META_DB ] 
        set_col_used_idx    = set()
        for c_key, cn, vec_cp in vec_col_pointers:
            if c_key != None and not total_results[ c_key ]:
                continue

            for cp in vec_cp:
                for cidx in total_results[ cp ][ :total_results[ cn ] ]:
                    if isinstance( cidx, list ):
                        set_col_used_idx.add( cidx[0] )
                    else:
                        set_col_used_idx.add( cidx )
        vec_col_used_idx    = list( set_col_used_idx )
        if len( vec_col_used_idx ) == 0 or 0 in vec_col_used_idx:
            return total_results

        tbl_used_idx    = set()
        for cidx in vec_col_used_idx:
            if db.vec_cols[ cidx ].table_belong != None:
                tbl_used_idx.add( db.vec_cols[ cidx ].table_belong.tbl_idx )
      
        total_results[ TV_TABLES_USED_IDX ] = list( tbl_used_idx )
        total_results[ TV_TABLES_NUM ]      = len( total_results[ TV_TABLES_USED_IDX ] )

        return total_results

    # Test on the features.
    def do_actual_test( self, sess, sg, target_data, batch_size, print_file = False, out_fn = None ):
        batch_num   = int( len( target_data ) / batch_size )
        if len( target_data ) % batch_size != 0:
            batch_num   += 1

        ti  = TestInfo()
        add_prop_test_info( ti, sg )
        add_tbl_test_info( ti, sg )
        ev_elist    = []

        if print_file:
            fout    = open( out_fn, "w" )

        pool    = ThreadPool( processes = 1 )
        prog        = Progbar( target = batch_num )
        for batch_idx in range( batch_num ):
            vec_data        = target_data[ batch_idx * batch_size: min( ( batch_idx + 1 ) * batch_size, len( target_data ) ) ]
            async_result    = pool.apply_async( threading_func, ( self, sess, sg, vec_data, ti ) )
            try:
                if batch_idx == 0:
                    return_val      = async_result.get( timeout = 60 )
                else:
                    return_val      = async_result.get( timeout = 5 )
            except:
                print ( "TIMEOUT." )
                return_val  = "SELECT TIMEOUT"
    
            vec_info    = return_val.split( "\t" )
            if len( vec_info ) < 2:
                ev_elist.append( [ return_val, "" ] )
            else:
                ev_elist.append( vec_info )

            if print_file:
                print ( return_val, file = fout )
            
            prog.update( batch_idx + 1 )

         # Evaluate.
        if print_file:
            em_f1   = evaluate_list( self._ev_glist, ev_elist, self._ev_dbdir, "match", self._ev_kmaps, print_file = fout )
            fout.close()
        else:
            em_f1   = evaluate_list( self._ev_glist, ev_elist, self._ev_dbdir, "match", self._ev_kmaps, print_diff = False )

        return em_f1

    # Merge subselect results to get one SQL statement.
    def generate_statement( self, vec_results ):
        cur_path    = [ PATH_NONE ]
        stat, _ = self._generate_select_statement( vec_results, cur_path, 1 )

        return stat

    def _generate_select_statement( self, vec_results, target_path, join_tbl_idx_start ):
        target_r    = None
        for r in vec_results:
            if r[ PF_PATH ] == target_path:
                target_r    = r
                break

        if target_r == None:
            return "", join_tbl_idx_start
            '''
            print ( "LOOKING PATH: ", target_path )
            for r in vec_results:
                for k, v in r.items():
                    print ( k, v )
                print ( "-----------------------------------------------------" )
            '''

        db  = target_r[ META_DB ]
        sql = ""

        # 1. Generate Table Join Statement.
        # tbl_name_map: table idx - its name ( like "T2" ). Empty if not used.
        tbl_statement, tbl_name_map, join_tbl_start = self._generate_tbl_join( db, target_r[ TV_TABLES_USED_IDX ], join_tbl_idx_start )
        if tbl_statement == "FAIL":
            target_r2    = self._get_used_table( target_r )
            tbl_statement, tbl_name_map, join_tbl_start = self._generate_tbl_join( db, target_r2[ TV_TABLES_USED_IDX ], join_tbl_idx_start )
            if tbl_statement == "FAIL":
                tbl_statement, tbl_name_map, join_tbl_start = self._generate_tbl_join( db, target_r[ TV_TABLES_USED_IDX ], join_tbl_idx_start, find_pkey_path = True )
            else:
                target_r    = target_r2


        # 2. Generate Select Statements.
        sel_statement   = self._generate_select( db, target_r, tbl_name_map )
        sql             = "SELECT %s FROM %s" % ( sel_statement, tbl_statement )
    
        # 3. WHERE statement.
        if target_r[ PF_WHERE ]:
            where_stat, join_tbl_idx_start  = self._generate_where( db, vec_results, target_r, tbl_name_map, target_path, join_tbl_idx_start )
            sql += " WHERE %s" % where_stat

        # 4. GROUP BY statement.
        if target_r[ PF_GROUPBY ]:
            sql += " GROUP BY %s" % self._generate_groupby( db, target_r[ GF_COLLIST ], tbl_name_map )
        
        # 5. HAVING statement.
        if target_r[ PF_HAVING ]:
            having_stat, join_tbl_idx_start  = self._generate_having( db, vec_results, target_r, tbl_name_map, target_path, join_tbl_idx_start )
            sql += " HAVING %s" % having_stat
        

        # 6. ORDER BY statements.
        if target_r[ PF_ORDERBY ]:
            sql += " ORDER BY %s" % self._generate_orderby( db, target_r, tbl_name_map )

        # 7. Limit statement.      
        if target_r[ PF_LIMIT ]:
            if target_r[ LF_ISMAX ]:
                sql += " LIMIT 1"
            else:
                limit_num   = target_r[ Q_BERT_TOK ][ target_r[ LF_POINTERLOC ] ]
                map_num_str = { 0: ["zero"], 1:["one", "single", "once" ], 2:["two", "twice"], 3:["three"], 4:["four"], 5:["five"], 6:["six"], 7:["seven"], 8:["eight"], 9:["nine"], 10:["ten" ] }   
                map_str_num = dict()
                for k, vec in map_num_str.items():
                    for v in vec:
                        map_str_num[v]  = k
                if limit_num.lower() in map_str_num:
                    limit_num   = map_str_num[ limit_num.lower() ]

                sql += " LIMIT %s" % str( limit_num )

        # 8. Union / Inter / Except.
        if target_path == [ PATH_NONE ] and target_r[ PF_MERGEOP ] > 0:
            if target_r[ PF_MERGEOP ] == 1:
                new_path    = [ PATH_UNION ]
                sql += " UNION "
            if target_r[ PF_MERGEOP ] == 2:
                new_path    = [ PATH_INTER ]
                sql += " INTERSECT "
            elif target_r[ PF_MERGEOP ] == 3:
                new_path    = [ PATH_EXCEPT ]
                sql += " EXCEPT "
            merge_stat, join_tbl_idx_start  = self._generate_select_statement( vec_results, new_path, join_tbl_idx_start )
            sql += merge_stat
        
        return sql, join_tbl_idx_start
        
    def _generate_orderby( self, db, target_r, tbl_name_map ):
        vu_num  = target_r[ OF_NUMVU ]
        vec_vus = []
        for idx in range( vu_num ):
            vu_stat, _ = self._generate_vu( db, tbl_name_map, \
                    target_r[ OF_VU_OPERATOR ][ idx ], target_r[ OF_VU_AGG1 ][ idx ], target_r[ OF_VU_COL1 ][ idx ], target_r[ OF_VU_DIST1 ][ idx ], \
                    target_r[ OF_VU_AGG2 ][ idx ], target_r[ OF_VU_COL2 ][ idx ], target_r[ OF_VU_DIST2 ][ idx ] )
            vec_vus.append( vu_stat )
        
        orderby_stat    = ", ".join( vec_vus )
        if target_r[ OF_DESCRIPTOR ] != 0:      # ASC: Default.
            orderby_stat    += " %s" %  IDX_INV_DESCRIPTORS[ target_r[ OF_DESCRIPTOR ] ].upper()
        return orderby_stat

    def _generate_groupby( self, db, vec_col_gb, tbl_name_map ):
        vec_gb    = []
        for c in vec_col_gb:
            if c == 0:
                continue
            tidx = db.vec_cols[ c ].table_belong.tbl_idx
            if tidx not in tbl_name_map:
                vec_gb.append( db.vec_cols[ c ].col_name_orig )
            else:
                vec_gb.append( "%s.%s" % ( tbl_name_map[ tidx ], db.vec_cols[ c ].col_name_orig ) )

        return  ",".join( vec_gb )

    def _generate_having( self, db, vec_results, target_r, tbl_name_map, cur_path, join_tbl_idx_start ):
        condunit_num    = target_r[ HV_NUM_CONDUNIT ]
        next_path       = cur_path.copy()
        if next_path == [ PATH_NONE ]:
            next_path   = [ PATH_HAVING ]
        else:
            next_path.append( PATH_HAVING )

        vec_cu_stats    = []
        for idx in range( condunit_num ):
            cu_statement, next_path, join_tbl_idx_start    = self._generate_condunit( db, vec_results, tbl_name_map, next_path, join_tbl_idx_start, target_r[ Q_BERT_TOK ],  \
                                                    target_r[ HV_CU_IS_NOT ][ idx ], target_r[ HV_CU_COND_OP ][ idx ], \
                                                    target_r[ HV_CU_VAL1_TYPE ][ idx ], target_r[ HV_CU_VAL1_SP ][ idx ], target_r[ HV_CU_VAL1_EP ][ idx ], target_r[ HV_CU_VAL1_LIKELY ][ idx ], target_r[ HV_CU_VAL1_BOOLVAL ][ idx ], \
                                                    target_r[ HV_CU_VAL2_TYPE ][ idx ], target_r[ HV_CU_VAL2_SP ][ idx ], target_r[ HV_CU_VAL2_EP ][ idx ], target_r[ HV_CU_VAL2_LIKELY ][ idx ], target_r[ HV_CU_VAL2_BOOLVAL ][ idx ], \
                                                    target_r[ HV_CU_VU_OPERATOR ][ idx ], target_r[ HV_CU_VU_AGG1 ][ idx ], target_r[ HV_CU_VU_COL1 ][ idx ], target_r[ HV_CU_VU_DIST1 ][ idx ], \
                                                    target_r[ HV_CU_VU_AGG2 ][ idx ], target_r[ HV_CU_VU_COL2 ][ idx ], target_r[ HV_CU_VU_DIST2 ][ idx ] )
            vec_cu_stats.append( cu_statement )

        where_stat  = ""
        for idx, cu_stat in enumerate( vec_cu_stats ):
            if idx == 0:
                where_stat  = cu_stat
            else:
                if target_r[ HV_CU_AGGREGATOR ][ idx ][0] == 0:
                    where_stat  += " and "
                else:
                    where_stat  += " or "
                where_stat  += cu_stat
        return where_stat, join_tbl_idx_start


    def _generate_where( self, db, vec_results, target_r, tbl_name_map, cur_path, join_tbl_idx_start ):
        condunit_num    = target_r[ WF_NUM_CONDUNIT ]
        next_path       = cur_path.copy()
        if next_path == [ PATH_NONE ]:
            next_path   = [ PATH_WHERE ]
        else:
            next_path.append( PATH_WHERE )

        vec_cu_stats    = []
        for idx in range( condunit_num ):
            cu_statement, next_path, join_tbl_idx_start    = self._generate_condunit( db, vec_results, tbl_name_map, next_path, join_tbl_idx_start, target_r[ Q_BERT_TOK ], \
                                                    target_r[ WF_CU_IS_NOT ][ idx ], target_r[ WF_CU_COND_OP ][ idx ], \
                                                    target_r[ WF_CU_VAL1_TYPE ][ idx ], target_r[ WF_CU_VAL1_SP ][ idx ], target_r[ WF_CU_VAL1_EP ][ idx ], target_r[ WF_CU_VAL1_LIKELY ][ idx ], target_r[ WF_CU_VAL1_BOOLVAL ][ idx ], \
                                                    target_r[ WF_CU_VAL2_TYPE ][ idx ], target_r[ WF_CU_VAL2_SP ][ idx ], target_r[ WF_CU_VAL2_EP ][ idx ], target_r[ WF_CU_VAL2_LIKELY ][ idx ], target_r[ WF_CU_VAL2_BOOLVAL ][ idx ], \
                                                    target_r[ WF_CU_VU_OPERATOR ][ idx ], target_r[ WF_CU_VU_AGG1 ][ idx ], target_r[ WF_CU_VU_COL1 ][ idx ], target_r[ WF_CU_VU_DIST1 ][ idx ], \
                                                    target_r[ WF_CU_VU_AGG2 ][ idx ], target_r[ WF_CU_VU_COL2 ][ idx ], target_r[ WF_CU_VU_DIST2 ][ idx ] )
            vec_cu_stats.append( cu_statement )

        where_stat  = ""
        for idx, cu_stat in enumerate( vec_cu_stats ):
            if idx == 0:
                where_stat  = cu_stat
            else:
                if target_r[ WF_CU_AGGREGATOR ][ idx ][0] == 0:
                    where_stat  += " and "
                else:
                    where_stat  += " or "
                where_stat  += cu_stat
        return where_stat, join_tbl_idx_start

    def _generate_condunit( self, db, vec_results, tbl_name_map, next_path, join_tbl_idx_start, q_toks, \
                                is_not, cond_op, val1_t, val1_sp, val1_ep, val1_likely, val1_b, val2_t, val2_sp, val2_ep, val2_likely, val2_b, \
                                vu_op, vu_agg1, vu_col1, vu_dist1, vu_agg2, vu_col2, vu_dist2 ):
        cu_stat, col_type = self._generate_vu( db, tbl_name_map, vu_op[0], vu_agg1[0], vu_col1[0], vu_dist1[0], vu_agg2[0], vu_col2[0], vu_dist2[0] )
        if is_not[0]:
            cu_stat = "%s NOT" % cu_stat
        cu_stat += " %s " % VEC_CONDOPS[ cond_op[0] ]

        val1_likely_fix = val1_likely[0]
        if cond_op[0] == 8 and val1_likely_fix == 0:    # LIKELY
            val1_likely_fix = val1_likely[1]

        val1_stat, next_path, join_tbl_idx_start   = self._generate_cu_val( db, vec_results, tbl_name_map, next_path, join_tbl_idx_start, \
                                            val1_t[0], val1_sp[0], val1_ep[0], val1_likely_fix, val1_b[0], q_toks, col_type )
        cu_stat += val1_stat

        if cond_op[0] == 0:    # Valid val2.
            val2_stat, next_path, join_tbl_idx_start   = self._generate_cu_val( db, vec_results, tbl_name_map, next_path, join_tbl_idx_start, \
                                            val2_t[0], val2_sp[0], val2_ep[0], 0, val2_b[0], q_toks, col_type )
            cu_stat += " AND %s" % val2_stat
        return cu_stat, next_path, join_tbl_idx_start

    def _generate_cu_val( self, db, vec_results, tbl_name_map, next_path, join_tbl_idx_start, \
                                t, sp, ep, likely, b, q_toks, col_type ):
        cu_val_stat = ""
        if t == 0:     # TEXT SPAN. 
            if ep < sp:
                ep = sp
            
            # GEnerate keyword.
            keyword = ""
            for t in q_toks[ sp:ep + 1 ]:
                if t[:2] == '##':
                    keyword += t[2:]
                else:
                    keyword += " " + t
            keyword = keyword.strip()            

            if len( keyword ) > 0 and keyword[0] == "'":
                keyword = keyword[1:]


            if likely == 1:
                keyword = "%" + keyword
            elif likely == 2:
                keyword = keyword + "%"
            elif likely == 3:
                keyword = "%" + keyword + "%"
            
            if col_type == "text":
                keyword = "'" + keyword + "'"
            elif col_type == "number":
                map_num_str = { 0: ["zero"], 1:["one", "single", "once" ], 2:["two", "twice"], 3:["three"], 4:["four"], 5:["five"], 6:["six"], 7:["seven"], 8:["eight"], 9:["nine"], 10:["ten" ] }   
                map_str_num = dict()
                for k, vec in map_num_str.items():
                    for v in vec:
                        map_str_num[v]  = k
                if keyword.lower() in map_str_num:
                    keyword = map_str_num[ keyword.lower() ]

            cu_val_stat += str( keyword )
        elif t == 1:   # BOOLEAN.
            if b:
                cu_val_stat += "T"
            else:
                cu_val_stat += "F"
        elif t == 2:   # Select statement.
            sel_stat, join_tbl_idx_start    = self._generate_select_statement( vec_results, next_path, join_tbl_idx_start )
            cu_val_stat     += "( %s )" % sel_stat
            next_path.append( PATH_PAR )
            next_path.append( PATH_WHERE )

        return cu_val_stat, next_path, join_tbl_idx_start
        

    def _generate_select( self, db, target_r, tbl_name_map ):
        condition_num   = target_r[ SF_NUM_VU ]
        vec_vu_info     = []
        for idx in range( condition_num ):
            vu_statement, _    = self._generate_vu( db, tbl_name_map, target_r[ SF_VU_OPERATOR ][ idx ], \
                                                target_r[ SF_VU_AGG1 ][ idx ], target_r[ SF_VU_COL1 ][ idx ], target_r[ SF_VU_DIST1 ][ idx ], \
                                                target_r[ SF_VU_AGG2 ][ idx ], target_r[ SF_VU_COL2 ][ idx ], target_r[ SF_VU_DIST2 ][ idx ] )

            if target_r[ SF_VU_AGGALL ][ idx ] != 0:
                vu_statement    = "%s( %s )" % ( VEC_AGGREGATORS[ target_r[ SF_VU_AGGALL ][ idx ] ], vu_statement )
            vec_vu_info.append( vu_statement )

        sel_stat    = ",".join( vec_vu_info )

        if target_r[ SF_DISTINCT ]:
            sel_stat    = "DISTINCT %s" % sel_stat

        return sel_stat

    def _generate_col( self, db, tbl_name_map, agg, col, dist ):
        col_type    = None
        if col == 0:    # *.
            q_str       = "%s" % ( db.vec_cols[ col ].col_name_orig ) 
            col_type    = "number"
        else:
            tbl_idx     = db.vec_cols[ col ].table_belong.tbl_idx
            col_type    = db.vec_cols[ col ].col_type
            if len( tbl_name_map ) == 0:
                q_str   = "%s" % ( db.vec_cols[ col ].col_name_orig ) 
            else:
                q_str   = "%s.%s" % ( tbl_name_map[ tbl_idx ], db.vec_cols[ col ].col_name_orig ) 

        if agg != 0:
            q_str   = "%s( %s )" % ( VEC_AGGREGATORS[ agg ], q_str )
        
        if dist:
            q_str   = "DISTINCT %s" % q_str
            
        return q_str, col_type

    def _generate_vu( self, db, tbl_name_map, vu_op, vu_agg1, vu_col1, vu_dist1, vu_agg2, vu_col2, vu_dist2 ):
        stat, col_type    = self._generate_col( db, tbl_name_map, vu_agg1, vu_col1, vu_dist1 )
        if vu_op != 0:
            stat    = "%s %s %s" % ( stat, VEC_OPERATORS[ vu_op ], self._generate_col( db, tbl_name_map, vu_agg2, vu_col2, vu_dist2 )[0] )

        return stat, col_type

        
    def _generate_tbl_join( self, db, vec_tbl_idx, join_tbl_idx_start, find_pkey_path = False ):
        if len( vec_tbl_idx ) == 1:
            return db.vec_tbls[ vec_tbl_idx[0] ].tbl_name_orig, dict(), join_tbl_idx_start

        # 1. Find the "Join Path". 
        joinable_tbls   = dict()
        join_fkey       = []
        for f1, f2 in db.foreign_keys:
            t1  = f1.table_belong.tbl_idx
            t2  = f2.table_belong.tbl_idx
            if t1 not in joinable_tbls:
                joinable_tbls[ t1 ] = set()
            if t2 not in joinable_tbls:
                joinable_tbls[ t2 ] = set()
            joinable_tbls[ t1 ].add( t2 )
            joinable_tbls[ t2 ].add( t1 )

            join_fkey.append( [ [ t1, t2 ], [ f1.col_idx, f2.col_idx ] ] )
            join_fkey.append( [ [ t2, t1 ], [ f2.col_idx, f1.col_idx ] ] )

        if find_pkey_path:
            # Tries to find additional links between primary keys.
            for tidx, t1 in enumerate( db.vec_tbls ):
                for t2 in db.vec_tbls[ tidx + 1: ]:
                    for c1 in t1.vec_primary_keys:
                        for c2 in t2.vec_primary_keys:
                            if ( ' '.join( c2.get_rep_name( cn_type = 0 ) ) in ' '.join( c1.get_rep_name( cn_type = 1 ) ) ) or \
                               ( ' '.join( c1.get_rep_name( cn_type = 0 ) ) in ' '.join( c2.get_rep_name( cn_type = 1 ) ) ):
                                join_fkey.append( [ [ t1.tbl_idx, t2.tbl_idx ], [ c1.col_idx, c2.col_idx ] ] )
                                join_fkey.append( [ [ t2.tbl_idx, t1.tbl_idx ], [ c2.col_idx, c1.col_idx ] ] )
                                if t1.tbl_idx not in joinable_tbls:
                                    joinable_tbls[ t1.tbl_idx ]  = set()
                                if t2.tbl_idx not in joinable_tbls:
                                    joinable_tbls[ t2.tbl_idx ]  = set()
                                joinable_tbls[ t1.tbl_idx ].add( t2.tbl_idx )
                                joinable_tbls[ t2.tbl_idx ].add( t1.tbl_idx )
       
        cur_path_cand   = [ [ set( [ v ] ), [ ] ] for v in vec_tbl_idx ]
        final_path      = None
        iter_num        = 0
        while len( cur_path_cand ) > 0 and iter_num < 6:
            iter_num    += 1
            new_path_cand   = []
            for path_cand, path_history in cur_path_cand:
                for t in path_cand:
                    if t not in joinable_tbls:
                        continue
                    for new_exp in joinable_tbls[ t ]:
                        if new_exp not in path_cand:
                            new_path            = path_cand.copy()
                            new_path_history    = path_history.copy()
                            new_path.add( new_exp )
                            new_path_history.append( [ t, new_exp ] )

                            if set( vec_tbl_idx ) <= new_path:
                                final_path  = [ new_path, new_path_history ]
                                break

                            if new_path in [ v[0] for v in cur_path_cand ]:
                                continue

                            if new_path not in [ v[0] for v in new_path_cand ]:
                                new_path_cand.append( [ new_path, new_path_history ] )
                                if len( new_path_cand ) > 100:
                                    break
                    if final_path != None:
                        break
                    if len( new_path_cand ) > 100:
                        break
                if final_path != None:
                    break
                if len( new_path_cand ) > 100:
                    break
            if final_path != None:
                break

            cur_path_cand   = new_path_cand

        if final_path == None:
            return "FAIL", dict(), join_tbl_idx_start

        # 2. Generate statements.
        tbl_name_map    = dict()
        statement       = None
        for t_join_1, t_join_2 in final_path[1]:
            if t_join_1 not in tbl_name_map:
                tbl_name_map[ t_join_1 ]    = "T%d" % join_tbl_idx_start
                join_tbl_idx_start  += 1
            if t_join_2 not in tbl_name_map:
                tbl_name_map[ t_join_2 ]    = "T%d" % join_tbl_idx_start
                join_tbl_idx_start  += 1

            # Find the applied foreign keys.
            fkey    = None
            for fkey_info in join_fkey:
                if fkey_info[0] == [ t_join_1, t_join_2 ]:
                    fkey    = fkey_info[1]
                    break
            
            if statement == None:
                statement   = "%s AS %s" % ( db.vec_tbls[ t_join_1 ].tbl_name_orig, tbl_name_map[ t_join_1 ] )
            statement   += " JOIN %s AS %s ON %s.%s = %s.%s" % ( db.vec_tbls[ t_join_2 ].tbl_name_orig, tbl_name_map[ t_join_2 ], tbl_name_map[ t_join_1 ], db.vec_cols[ fkey[0] ].col_name_orig, tbl_name_map[ t_join_2 ], db.vec_cols[ fkey[1] ].col_name_orig )
    
        return statement, tbl_name_map, join_tbl_idx_start
                

    def test( self, load_path, batch_size, out_fn ):
        sg = SQLGen( bert_config = self._bert_config  )
        sg.constructGraph( )
        
        config                          = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement     = True
        with tf.Session( graph = sg.graph, config = config ) as sess:
            sg.init.run()
            self._load_model( sess, sg, load_path )
            self.do_actual_test( sess, sg, self.vec_prop_valid, batch_size, print_file = True, out_fn = out_fn )
    
    def reduce_size( self, load_path  ):
        sg = SQLGen( bert_config = self._bert_config  )
        sg.constructGraph( )
        
        config                          = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement     = True
        with tf.Session( graph = sg.graph, config = config ) as sess:
            sg.init.run()
            self._load_model( sess, sg, load_path )
            sg.saver.save( sess, "reduced/best_chk"  )


    def _load_model( self, sess, sg, save_path ):
        ckpt    = tf.train.get_checkpoint_state( save_path )
        v2_path = ckpt.model_checkpoint_path + ".index" if ckpt else ""

        sg.saver.restore( sess, ckpt.model_checkpoint_path )

if __name__== "__main__":
    if len( sys.argv ) < 4:
        print ( "Usage: python actual_test.py [MODEL_PATH] [BERT_CONFIG_DIR] [INPUT_DIR] [OUTPUT_FN]" )
        sys.exit(0)

    model_path      = sys.argv[1]
    BERT_DIR        = sys.argv[2]
    input_dir       = sys.argv[3]
    out_path        = sys.argv[4] 

    sys.path.append( input_dir )
    from evaluation         import *
    bert_tokenizer  = tokenization.FullTokenizer( vocab_file = "%s/vocab.txt" % BERT_DIR, do_lower_case = True )
    bert_config     = modeling.BertConfig.from_json_file( "%s/bert_config.json" % BERT_DIR )
    st  = SQLTester( "%s/dev_gold.sql" % input_dir, "%s/database" % input_dir, "%s/tables.json" % input_dir, bert_tokenizer = bert_tokenizer, bert_config = bert_config )
    st.readCorpus( "%s/tables.json" % input_dir, "%s/dev.json" % input_dir )

    st.test( model_path, 1, out_path )



