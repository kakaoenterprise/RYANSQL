import sys
import numpy as np

from padding_func   import *
from db_meta        import *

def extract_validcols( prop_data, valid_tbl_idx, inference = False ):
    vec_update_targets_answers  = [ GF_COLLIST, OF_VU_COL1, OF_VU_COL2, SF_VU_COL1, SF_VU_COL2, WF_CU_VU_COL1, WF_CU_VU_COL2, HV_CU_VU_COL1, HV_CU_VU_COL2 ]    # Updated for training; in valid, anyway those values are not used.

    db  = prop_data[ META_DB ]
    prop_data_ret   = dict()
    for k, v in prop_data.items():
        prop_data_ret[ k ]  = v     # Value copy.

    # 1. Get the valid column span. ( Inclusive )
    vec_col_spans   = [ [ 0, 0 ] ]  # Special colum: *
    for t in valid_tbl_idx:
        if t < len( db.vec_tbls ):
            tbl = db.vec_tbls[ t ]
            vec_col_spans.append( [ tbl.vec_cols[0].col_idx, tbl.vec_cols[-1].col_idx ] )
        else:
            print ( "ERROR: %d VS. %d" % ( t, len( db.vec_tbls ) ) )
    vec_col_spans   = sorted( vec_col_spans, key = lambda x: x[0] )

    # 2. Generate Col Idx - New Col Idx Map.
    new_idx     = 0
    map_col_idx = dict()
    for sidx, eidx in vec_col_spans:
        for col_idx in range( sidx, eidx + 1 ):
            map_col_idx[ col_idx ]  = new_idx
            new_idx += 1
    
    # 4. Update col idx of the answers.
    if inference == False:
        for a_key in vec_update_targets_answers:
            prop_data_ret[ a_key ]  = [ map_col_idx[ col_idx ] for col_idx in prop_data[ a_key ] if col_idx in map_col_idx ]

    return prop_data_ret

# Generate feed dict elements for BERT.
def _generate_bert_feed( vec_prop_data, bert_tokenizer, inference = False, is_prop = False, vec_valid_tbl = None ):
    vec_bert_id         = []
    vec_bert_segs       = []
    vec_col_loc         = []
    vec_tbl_col_idx     = []
    for v_idx, v_f in enumerate( vec_prop_data ):
        vec_bert_tok    = [ "[CLS]" ] + v_f[ Q_BERT_TOK ] + [ "[SEP]" ]

        # ADD PATH.
        vec_bert_tok    += [ t.lower() for t in v_f[ PF_PATH ] ] + [ "[SEP]" ]
        bert_seg_id     = [0] * len( vec_bert_tok )

        col_loc         = []
        db_col_idx     = []
        col_idx         = 0
        if is_prop: # Add the "*".
            vec_bert_tok    += [ "*", "[SEP]" ]
            col_loc.append( len( vec_bert_tok ) - 2 )
            bert_seg_id     += [1] * 2
            col_idx     += 1

        for tbl_idx, vec_bert_tbl in enumerate( v_f[ C_BERT_TOK ] ):
            if is_prop and vec_valid_tbl != None and tbl_idx not in vec_valid_tbl[ v_idx ]:  # Ignore the tbl.
                continue

            for vec_bert_col in vec_bert_tbl:
                vec_bert_tok    += vec_bert_col + [ "[SEP]" ]
                col_loc.append( len( vec_bert_tok ) - 2 )
                bert_seg_id += [1] * ( len( vec_bert_col ) + 1 )
            db_col_idx.append( [ idx + col_idx for idx in range( len( vec_bert_tbl ) ) ] )
            col_idx += len( vec_bert_tbl )

        if len( vec_bert_tok ) >= 512:
            vec_bert_tok    = vec_bert_tok[ :512 ]
            bert_seg_id     = bert_seg_id[ :512 ]
            col_loc         = [ v for v in col_loc if v < 512 ]
            db_col_idx      = [ [ v for v in v2 if v < 512 ] for v2 in db_col_idx ]
        
        vec_bert_id.append( bert_tokenizer.convert_tokens_to_ids( vec_bert_tok ) )        
        vec_bert_segs.append( bert_seg_id )
        vec_col_loc.append( col_loc )
        vec_tbl_col_idx.append( db_col_idx )

    # MASKS.
    vec_bert_mask       = [ len(v) for v in vec_bert_id ]
    vec_q_mask          = [ len( v_f[ Q_BERT_TOK ] ) for v_f in vec_prop_data ]
    vec_col_mask        = [ len(v) for v in vec_col_loc ]
    vec_tbl_col_mask    = [ [ len( tbl_col_idx ) for tbl_col_idx in db_col_idx ] for db_col_idx in vec_tbl_col_idx ]
    vec_tbl_mask        = [ len( db_col_idx ) for db_col_idx in vec_tbl_col_idx ]

    # Do Padding.
    vec_bert_id         = padding_2D( vec_bert_id, 0 )[0]
    vec_bert_segs       = padding_2D( vec_bert_segs, 0 )[0]
    vec_col_loc         = padding_2D( vec_col_loc, 0 )[0]

    if not is_prop:
        vec_tbl_col_idx     = padding_3D_to_2D( vec_tbl_col_idx, 0 )[0]
        vec_tbl_col_mask    = padding_2D( vec_tbl_col_mask, 0 )[0]

    # INSERT INTO DIC.
    bert_feed   = dict()
    bert_feed[ "%s:0" % BERT_ID ]   = vec_bert_id
    bert_feed[ "bert_mask:0" ]      = vec_bert_mask
    bert_feed[ "bert_seg_id:0" ]    = vec_bert_segs
    bert_feed[ "bert_q_mask:0" ]    = vec_q_mask 
    bert_feed[ "bert_col_loc:0" ]   = vec_col_loc
    bert_feed[ "bert_col_mask:0" ]  = vec_col_mask
    if not is_prop:
        bert_feed[ "bert_tbl_col_idx:0" ]   = vec_tbl_col_idx
        bert_feed[ "bert_tbl_col_mask:0" ]  = vec_tbl_col_mask
        bert_feed[ "bert_tbl_mask:0" ]  = vec_tbl_mask

    return bert_feed

# vec_prop_data: vector of feature dictionaries.
# vec_valid_tbl_idx: vector of valid table indexes. Should have the same batch size.
def prepare_prop_dict( vec_prop_data, vec_valid_tbl_idx, inference = False, bert_tokenizer = None ):
    if len( vec_prop_data ) != len( vec_valid_tbl_idx ):
        print ( "LENGTH MISMATCH: [%d] VS. [%d]" % ( len( vec_prop_data ), len( vec_valid_tbl_idx ) ) )
        sys.exit(0)

    vec_updated_prop_data   = []
    for prop_data, valid_tbl_idx in zip( vec_prop_data, vec_valid_tbl_idx ):
        prop_data_flatten   = dict()
        for k, v in prop_data.items():
            prop_data_flatten[k]    = v

        vec_updated_prop_data.append( extract_validcols( prop_data_flatten, valid_tbl_idx, inference = inference ) )
    vec_prop_data   = vec_updated_prop_data


    if inference:
        vec_extraction_targets  = [ PF_PATHIDX ]
        vec_paddings            = []
    else:
        vec_extraction_targets  = [ PF_PATHIDX, \
                                PF_MERGEOP, PF_FROMSQL, PF_ORDERBY, PF_GROUPBY, PF_LIMIT, PF_WHERE, PF_HAVING, \
                                GF_NUMCOL, GF_COLLIST, \
                                OF_NUMVU, OF_DESCRIPTOR, OF_VU_COL1, OF_VU_COL2, OF_VU_AGG1, OF_VU_AGG2, OF_VU_DIST1, OF_VU_DIST2, OF_VU_OPERATOR, \
                                LF_ISMAX, LF_POINTERLOC, \
                                SF_NUM_VU, SF_DISTINCT, SF_VU_AGGALL, SF_VU_COL1, SF_VU_COL2, SF_VU_AGG1, SF_VU_AGG2, SF_VU_DIST1, SF_VU_DIST2, SF_VU_OPERATOR, \
                                WF_NUM_CONDUNIT, WF_CU_AGGREGATOR, WF_CU_IS_NOT, WF_CU_COND_OP, \
                                WF_CU_VAL1_TYPE, WF_CU_VAL1_SP, WF_CU_VAL1_EP, WF_CU_VAL1_LIKELY, WF_CU_VAL1_BOOLVAL, \
                                WF_CU_VAL2_TYPE, WF_CU_VAL2_SP, WF_CU_VAL2_EP, WF_CU_VAL2_LIKELY, WF_CU_VAL2_BOOLVAL, \
                                WF_CU_VU_OPERATOR, WF_CU_VU_AGG1, WF_CU_VU_COL1, WF_CU_VU_DIST1, WF_CU_VU_AGG2, WF_CU_VU_COL2, WF_CU_VU_DIST2, WF_CU_VAL1_IGNORE, WF_CU_VAL2_IGNORE, \
                                HV_NUM_CONDUNIT, HV_CU_AGGREGATOR, HV_CU_IS_NOT, HV_CU_COND_OP, \
                                HV_CU_VAL1_TYPE, HV_CU_VAL1_SP, HV_CU_VAL1_EP, HV_CU_VAL1_LIKELY, HV_CU_VAL1_BOOLVAL, \
                                HV_CU_VAL2_TYPE, HV_CU_VAL2_SP, HV_CU_VAL2_EP, HV_CU_VAL2_LIKELY, HV_CU_VAL2_BOOLVAL, \
                                HV_CU_VU_OPERATOR, HV_CU_VU_AGG1, HV_CU_VU_COL1, HV_CU_VU_DIST1, HV_CU_VU_AGG2, HV_CU_VU_COL2, HV_CU_VU_DIST2, HV_CU_VAL1_IGNORE, HV_CU_VAL2_IGNORE ]

        vec_paddings = [ [ 3, [ GF_COLLIST ] ], \
                     [ 3, [ OF_VU_COL1, OF_VU_COL2, OF_VU_AGG1, OF_VU_AGG2, OF_VU_DIST1, OF_VU_DIST2, OF_VU_OPERATOR ] ], \
                     [ 6, [ SF_VU_AGGALL, SF_VU_COL1, SF_VU_COL2, SF_VU_AGG1, SF_VU_AGG2, SF_VU_DIST1, SF_VU_DIST2, SF_VU_OPERATOR ] ], \
                     [ 4, [ WF_CU_AGGREGATOR, WF_CU_IS_NOT, WF_CU_COND_OP, WF_CU_VAL1_TYPE, WF_CU_VAL1_SP, WF_CU_VAL1_EP, WF_CU_VAL1_LIKELY, WF_CU_VAL1_BOOLVAL, \
                            WF_CU_VAL2_TYPE, WF_CU_VAL2_SP, WF_CU_VAL2_EP, WF_CU_VAL2_LIKELY, WF_CU_VAL2_BOOLVAL, \
                            WF_CU_VU_OPERATOR, WF_CU_VU_AGG1, WF_CU_VU_COL1, WF_CU_VU_DIST1, WF_CU_VU_AGG2, WF_CU_VU_COL2, WF_CU_VU_DIST2, WF_CU_VAL1_IGNORE, WF_CU_VAL2_IGNORE ] ],
                     [ 2, [ HV_CU_AGGREGATOR, HV_CU_IS_NOT, HV_CU_COND_OP, HV_CU_VAL1_TYPE, HV_CU_VAL1_SP, HV_CU_VAL1_EP, HV_CU_VAL1_LIKELY, HV_CU_VAL1_BOOLVAL, \
                            HV_CU_VAL2_TYPE, HV_CU_VAL2_SP, HV_CU_VAL2_EP, HV_CU_VAL2_LIKELY, HV_CU_VAL2_BOOLVAL, \
                            HV_CU_VU_OPERATOR, HV_CU_VU_AGG1, HV_CU_VU_COL1, HV_CU_VU_DIST1, HV_CU_VU_AGG2, HV_CU_VU_COL2, HV_CU_VU_DIST2, HV_CU_VAL1_IGNORE, HV_CU_VAL2_IGNORE ] ] ]

    # Special vars.
    vec_cur_path_main   = []
    vec_cur_path_mask   = []

    # Init & Extract Features.
    map_feature_vecs    = dict()
    for ext_key in vec_extraction_targets:
        map_feature_vecs[ ext_key ] = []

    for v_f in vec_prop_data:
        for ext_key in vec_extraction_targets:
            map_feature_vecs[ ext_key ].append( v_f[ ext_key ] )

        if v_f[ PF_PATH ][0] == PATH_NONE:
            vec_cur_path_main.append( True )
        else:
            vec_cur_path_main.append( False )

    # Do Padding.
    vec_cur_path_mask   = [ len( p ) for p in map_feature_vecs[ PF_PATHIDX ] ]

    # Padding - Special cases.
    map_feature_vecs[ PF_PATHIDX ]      = padding_2D( map_feature_vecs[ PF_PATHIDX ], 0 )[0]
    for pad_size, key_list in vec_paddings:
        for key in key_list:
            map_feature_vecs[ key ]     = padding_2D_len( map_feature_vecs[ key ], pad_size, 0 )

    # Generate Dictionary.
    # 1. Special cases.
    feed_dict = { \
        "cur_path_main:0" : vec_cur_path_main, \
        "cur_path_mask:0" : vec_cur_path_mask, \
        "is_train:0" : False, \
        "drop_rate:0" : 0.0 }
    
    # 2. Others.
    for ext_key in vec_extraction_targets:
        feed_dict[ "%s:0" % ext_key ]   = map_feature_vecs[ ext_key ]
    
    bert_feed   = _generate_bert_feed( vec_prop_data, bert_tokenizer = bert_tokenizer, inference = inference, is_prop = True, vec_valid_tbl = vec_valid_tbl_idx )
    for k, v in bert_feed.items():
        feed_dict[ k ]  = v

    return feed_dict

# vec_prop_data: vector of feature dictionaries.
def prepare_tbl_dict( vec_prop_data, inference = False, bert_tokenizer = None ):
    vec_extraction_targets  = [ PF_PATHIDX ]
    if inference == False:
        vec_extraction_targets  += [ TV_TABLES_NUM, TV_TABLES_USED_IDX ]

    # Special vars.
    vec_cur_path_main   = []
    vec_cur_path_mask   = []

    # Init & Extract Features.
    map_feature_vecs    = dict()
    for ext_key in vec_extraction_targets:
        map_feature_vecs[ ext_key ] = []

    vec_tot_tbl_num     = []
    for v_f in vec_prop_data:
        for ext_key in vec_extraction_targets:
            map_feature_vecs[ ext_key ].append( v_f[ ext_key ] )

        if v_f[ PF_PATH ][0] == PATH_NONE:
            vec_cur_path_main.append( True )
        else:
            vec_cur_path_main.append( False )
        vec_tot_tbl_num.append( len( v_f[ TV_TABLES_NAME ] ) )

    # Do Padding.
    vec_cur_path_mask   = [ len( p ) for p in map_feature_vecs[ PF_PATHIDX ] ]

    # Padding - Special cases.
    map_feature_vecs[ PF_PATHIDX ]      = padding_2D( map_feature_vecs[ PF_PATHIDX ], 0 )[0]
    if not inference:
        map_feature_vecs[ TV_TABLES_USED_IDX ]  = [ make_one_hot( v, max( vec_tot_tbl_num ) ) for v in map_feature_vecs[ TV_TABLES_USED_IDX ] ]
    
    # Generate Dictionary.
    # 1. Special cases.
    feed_dict = { \
        "cur_path_main:0" : vec_cur_path_main, \
        "cur_path_mask:0" : vec_cur_path_mask, \
        "is_train:0" : False, \
        "drop_rate:0" : 0.0 }
    
    # 2. Others.
    for ext_key in vec_extraction_targets:
        feed_dict[ "%s:0" % ext_key ]   = map_feature_vecs[ ext_key ]

    bert_feed   = _generate_bert_feed( vec_prop_data, bert_tokenizer = bert_tokenizer, inference = inference )
    for k, v in bert_feed.items():
        feed_dict[ k ]  = v

    return feed_dict


def add_prop_test_info( ti, sg ):
    def _arg2_test( data, result, key_comp, key_mark, key_num ):
        result  = result[ key_comp ]
        if len( result ) != len( data[ key_comp ] ):
            return False

        for e, d, r in zip( data[ key_mark ], data[ key_comp ], result ):
            if e == 0:  # INVALID DATA
                continue

            if d != r:
                return False
        return True
    
    # Evaluate only when the value of data[ mask ] is equal to mask_val.
    def _masked_match( data, result, KEY_VAL, KEY_LEN, v_mask = None, v_mask_val = None ):
        source_list = data[ KEY_VAL ]
        target_list = result[ KEY_VAL ]
        if v_mask == None:
            return source_list == target_list

        if len( source_list ) != len( target_list ):
            return False

        for idx, s in enumerate( source_list ):
            check   = True
            for mask, mask_val in zip( v_mask, v_mask_val ):
                if data[ mask ][ idx ] != mask_val:
                    check   = False

            if check:
                if s != target_list[ idx ]:
                    return False
       
        return True
        
    ti.add_test_val( PF_MERGEOP, sg.merge_op_result, "MERGE_OP", cond = lambda data, result: data[ PF_PATH ][0] == PATH_NONE )
    ti.add_test_val( PF_FROMSQL, sg.from_sql_result, "FROM_SQL", cond = lambda data, result: data[ PF_PATH ][0] == PATH_NONE )
    ti.add_test_val( PF_ORDERBY, sg.order_by_result, "ORDER_BY" )
    ti.add_test_val( PF_GROUPBY, sg.group_by_result, "GROUP_BY" )
        
    ti.add_test_val( PF_MERGEOP, sg.merge_op_result, "MERGE_OP", cond = lambda data, result: data[ PF_PATH ][0] == PATH_NONE )
    ti.add_test_val( PF_FROMSQL, sg.from_sql_result, "FROM_SQL", cond = lambda data, result: data[ PF_PATH ][0] == PATH_NONE )
    ti.add_test_val( PF_ORDERBY, sg.order_by_result, "ORDER_BY" )
    ti.add_test_val( PF_GROUPBY, sg.group_by_result, "GROUP_BY" )
    ti.add_test_val( PF_LIMIT, sg.limit_result, "LIMIT" )
    ti.add_test_val( PF_WHERE, sg.where_result, "WHERE" )
    ti.add_test_val( PF_HAVING, sg.having_result, "HAVING" )
    ti.add_test_val( GF_NUMCOL, sg.gb_num_result, "GB_COL_NUM", cond = lambda data, result: data[ PF_GROUPBY ] == 1 )
    ti.add_test_val( GF_COLLIST, sg.gb_col_result, "GB_COL_LIST", cond = lambda data, result: data[ PF_GROUPBY ] == 1, match_func = lambda data, result: set( data[ GF_COLLIST ] ) == set( result[ GF_COLLIST ] ) )

    ti.add_test_val( OF_NUMVU, sg.ob_num_result, "OB_COL_NUM", cond = lambda data, result: data[ PF_ORDERBY ] == 1 )
    ti.add_test_val( OF_DESCRIPTOR, sg.ob_desc_result, "OB_DESC", cond = lambda data, result: data[ PF_ORDERBY ] == 1 )
    ti.add_test_val( OF_VU_OPERATOR, sg.ob_op_r, "OB_OPERATOR", cond = lambda data, result: data[ PF_ORDERBY ] == 1 )
    ti.add_test_val( OF_VU_COL1, sg.ob_p1_r, "OB_COL1", cond = lambda data, result: data[ PF_ORDERBY ] == 1 )
    ti.add_test_val( OF_VU_AGG1, sg.ob_agg1_r, "OB_AGG1", cond = lambda data, result: data[ PF_ORDERBY ] == 1 )
    ti.add_test_val( OF_VU_DIST1, sg.ob_dist1_r, "OB_DIST1", cond = lambda data, result: data[ PF_ORDERBY ] == 1 )
    ti.add_test_val( OF_VU_COL2, sg.ob_p2_r, "OB_COL2", cond = lambda data, result: data[ PF_ORDERBY ] == 1 and max( data[ OF_VU_OPERATOR ] ) > 0, match_func = lambda data, result: _arg2_test( data, result, OF_VU_COL2, OF_VU_OPERATOR, OF_NUMVU ) )
    ti.add_test_val( OF_VU_AGG2, sg.ob_agg2_r, "OB_AGG2", cond = lambda data, result: data[ PF_ORDERBY ] == 1 and max( data[ OF_VU_OPERATOR ] ) > 0, match_func = lambda data, result: _arg2_test( data, result, OF_VU_AGG2, OF_VU_OPERATOR, OF_NUMVU ) )
    ti.add_test_val( OF_VU_DIST2, sg.ob_dist2_r, "OB_DIST2", cond = lambda data, result: data[ PF_ORDERBY ] == 1 and max( data[ OF_VU_OPERATOR ] ) > 0, match_func = lambda data, result: _arg2_test( data, result, OF_VU_DIST2, OF_VU_OPERATOR, OF_NUMVU ) )

    ti.add_test_val( LF_ISMAX, sg.lm_ismax_r, "LM_ISMAX", cond = lambda data, result: data[ PF_LIMIT ] == 1 )
    ti.add_test_val( LF_POINTERLOC, sg.lm_maxptr_r, "LM_LOC", cond = lambda data, result: data[ PF_LIMIT ] == 1 and data[ LF_ISMAX ] == 0 )

    ti.add_test_val( SF_NUM_VU, sg.sel_num_r, "SEL_COL_NUM" )
    ti.add_test_val( SF_DISTINCT, sg.sel_dist_r, "SEL_DISTINCT" )
    ti.add_test_val( SF_VU_OPERATOR, sg.sel_op_r, "SEL_OPERATOR" )
    ti.add_test_val( SF_VU_AGGALL, sg.sel_aggtot_r, "SEL_AGGTOT" )
    ti.add_test_val( SF_VU_COL1, sg.sel_p1_r, "SEL_COL1" )
    ti.add_test_val( SF_VU_AGG1, sg.sel_agg1_r, "SEL_AGG1" )
    ti.add_test_val( SF_VU_DIST1, sg.sel_dist1_r, "SEL_DIST1" )
    ti.add_test_val( SF_VU_COL2, sg.sel_p2_r, "SEL_COL2", cond = lambda data, result: max( data[ SF_VU_OPERATOR ] ) > 0, match_func = lambda data, result: _arg2_test( data, result, SF_VU_COL2, SF_VU_OPERATOR, SF_NUM_VU ) )
    ti.add_test_val( SF_VU_AGG2, sg.sel_agg2_r, "SEL_AGG2", cond = lambda data, result: max( data[ SF_VU_OPERATOR ] ) > 0, match_func = lambda data, result: _arg2_test( data, result, SF_VU_AGG2, SF_VU_OPERATOR, SF_NUM_VU ) )
    ti.add_test_val( SF_VU_DIST2, sg.sel_dist2_r, "SEL_DIST2", cond = lambda data, result: max( data[ SF_VU_OPERATOR ] ) > 0, match_func = lambda data, result: _arg2_test( data, result, SF_VU_DIST2, SF_VU_OPERATOR, SF_NUM_VU ) )

    ti.add_test_val( WF_NUM_CONDUNIT, sg.where_num_r, "WHERE_COND_NUM", cond = lambda data, result: data[ PF_WHERE ] == 1 )
    ti.add_test_val( WF_CU_AGGREGATOR, sg.wh_agg_s, "WHERE_AGGREGATOR", cond = lambda data, result: data[ PF_WHERE ] == 1 )
    ti.add_test_val( WF_CU_IS_NOT, sg.wh_isnot_s, "WHERE_ISNOT", cond = lambda data, result: data[ PF_WHERE ] == 1 )
    ti.add_test_val( WF_CU_COND_OP, sg.wh_condop_s, "WHERE_CONDOP", cond = lambda data, result: data[ PF_WHERE ] == 1 )
    ti.add_test_val( WF_CU_VAL1_TYPE, sg.wh_val1_type_s, "WHERE_VAL1_TYPE", cond = lambda data, result: data[ PF_WHERE ] == 1 )
    ti.add_test_val( WF_CU_VAL1_LIKELY, sg.wh_val1_like_s, "WHERE_VAL1_LIKE", cond = lambda data, result: data[ PF_WHERE ] == 1 and 0 in data[ WF_CU_VAL1_TYPE ], match_func = lambda data, result: _masked_match( data, result, WF_CU_VAL1_LIKELY, WF_NUM_CONDUNIT, [ WF_CU_VAL1_TYPE ], [ 0 ] ) )
    ti.add_test_val( WF_CU_VAL1_BOOLVAL, sg.wh_val1_bool_s, "WHERE_VAL1_BOOL", cond = lambda data, result: data[ PF_WHERE ] == 1 and 1 in data[ WF_CU_VAL1_TYPE ], match_func = lambda data, result: _masked_match( data, result, WF_CU_VAL1_BOOLVAL, WF_NUM_CONDUNIT, [ WF_CU_VAL1_TYPE ], [ 1 ] ) )
    ti.add_test_val( WF_CU_VAL1_SP, sg.wh_val1_sp_s, "WHERE_VAL1_SP", cond = lambda data, result: data[ PF_WHERE ] == 1 and 0 in data[ WF_CU_VAL1_TYPE ], match_func = lambda data, result: _masked_match( data, result, WF_CU_VAL1_SP, WF_NUM_CONDUNIT, [ WF_CU_VAL1_TYPE ], [ 0 ] ) )
    ti.add_test_val( WF_CU_VAL1_EP, sg.wh_val1_ep_s, "WHERE_VAL1_EP", cond = lambda data, result: data[ PF_WHERE ] == 1 and 0 in data[ WF_CU_VAL1_TYPE ], match_func = lambda data, result: _masked_match( data, result, WF_CU_VAL1_EP, WF_NUM_CONDUNIT, [ WF_CU_VAL1_TYPE ], [ 0 ] ) )
    
    ti.add_test_val( WF_CU_VAL2_TYPE, sg.wh_val2_type_s, "WHERE_VAL2_TYPE", cond = lambda data, result: data[ PF_WHERE ] == 1 and 0 in data[ WF_CU_COND_OP ], match_func = lambda data, result: _masked_match( data, result, WF_CU_VAL2_TYPE, WF_NUM_CONDUNIT, [ WF_CU_COND_OP ], [ 0 ] ) )
    ti.add_test_val( WF_CU_VAL2_LIKELY, sg.wh_val2_like_s, "WHERE_VAL2_LIKE", cond = lambda data, result: data[ PF_WHERE ] == 1 and 0 in data[ WF_CU_COND_OP ] and 0 in data[ WF_CU_VAL2_TYPE ], match_func = lambda data, result: _masked_match( data, result, WF_CU_VAL2_LIKELY, WF_NUM_CONDUNIT, [ WF_CU_COND_OP, WF_CU_VAL2_TYPE ], [ 0, 0 ] ) )
    ti.add_test_val( WF_CU_VAL2_BOOLVAL, sg.wh_val2_bool_s, "WHERE_VAL2_BOOL", cond = lambda data, result: data[ PF_WHERE ] == 1 and 0 in data[ WF_CU_COND_OP ] and 0 in data[ WF_CU_VAL2_TYPE ], match_func = lambda data, result: _masked_match( data, result, WF_CU_VAL2_BOOLVAL, WF_NUM_CONDUNIT, [ WF_CU_COND_OP, WF_CU_VAL2_TYPE ], [ 0, 1 ] ) )
    ti.add_test_val( WF_CU_VAL2_SP, sg.wh_val2_sp_s, "WHERE_VAL2_SP", cond = lambda data, result: data[ PF_WHERE ] == 1 and 0 in data[ WF_CU_COND_OP ] and 0 in data[ WF_CU_VAL2_TYPE ], match_func = lambda data, result: _masked_match( data, result, WF_CU_VAL2_SP, WF_NUM_CONDUNIT, [ WF_CU_COND_OP, WF_CU_VAL2_TYPE ], [ 0, 0 ] ) )
    ti.add_test_val( WF_CU_VAL2_EP, sg.wh_val2_ep_s, "WHERE_VAL2_EP", cond = lambda data, result: data[ PF_WHERE ] == 1 and 0 in data[ WF_CU_COND_OP ] and 0 in data[ WF_CU_VAL2_TYPE ], match_func = lambda data, result: _masked_match( data, result, WF_CU_VAL2_EP, WF_NUM_CONDUNIT, [ WF_CU_COND_OP, WF_CU_VAL2_TYPE ], [ 0, 0 ] ) )

    ti.add_test_val( WF_CU_VU_OPERATOR, sg.wh_vu_op_s, "WHERE_VU_OP", cond = lambda data, result: data[ PF_WHERE ] == 1 )
    ti.add_test_val( WF_CU_VU_AGG1, sg.wh_vu_agg1_s, "WHERE_VU_AGG1", cond = lambda data, result: data[ PF_WHERE ] == 1 )
    ti.add_test_val( WF_CU_VU_COL1, sg.wh_vu_col1_s, "WHERE_VU_COL1", cond = lambda data, result: data[ PF_WHERE ] == 1 )
    ti.add_test_val( WF_CU_VU_DIST1, sg.wh_vu_dist1_s, "WHERE_VU_DIST1", cond = lambda data, result: data[ PF_WHERE ] == 1 )
    ti.add_test_val( WF_CU_VU_AGG2, sg.wh_vu_agg2_s, "WHERE_VU_AGG2", cond = lambda data, result: data[ PF_WHERE ] == 1 and max( data[ WF_CU_VU_OPERATOR ] ) > 0, match_func = lambda data, result: _arg2_test( data, result, WF_CU_VU_AGG2, WF_CU_VU_OPERATOR, WF_NUM_CONDUNIT ) )
    ti.add_test_val( WF_CU_VU_COL2, sg.wh_vu_col2_s, "WHERE_VU_COL2", cond = lambda data, result: data[ PF_WHERE ] == 1 and max( data[ WF_CU_VU_OPERATOR ] ) > 0, match_func = lambda data, result: _arg2_test( data, result, WF_CU_VU_COL2, WF_CU_VU_OPERATOR, WF_NUM_CONDUNIT ) )
    ti.add_test_val( WF_CU_VU_DIST2, sg.wh_vu_dist2_s, "WHERE_VU_DIST2", cond = lambda data, result: data[ PF_WHERE ] == 1 and max( data[ WF_CU_VU_OPERATOR ] ) > 0, match_func = lambda data, result: _arg2_test( data, result, WF_CU_VU_DIST2, WF_CU_VU_OPERATOR, WF_NUM_CONDUNIT ) )

    ti.add_test_val( HV_NUM_CONDUNIT, sg.having_num_r, "HAVING_COND_NUM", cond = lambda data, result: data[ PF_HAVING ] == 1 )
    ti.add_test_val( HV_CU_AGGREGATOR, sg.hv_agg_s, "HAVING_AGGREGATOR", cond = lambda data, result: data[ PF_HAVING ] == 1 )
    ti.add_test_val( HV_CU_IS_NOT, sg.hv_isnot_s, "HAVING_ISNOT", cond = lambda data, result: data[ PF_HAVING ] == 1 )
    ti.add_test_val( HV_CU_COND_OP, sg.hv_condop_s, "HAVING_CONDOP", cond = lambda data, result: data[ PF_HAVING ] == 1 )
    ti.add_test_val( HV_CU_VAL1_TYPE, sg.hv_val1_type_s, "HAVING_VAL1_TYPE", cond = lambda data, result: data[ PF_HAVING ] == 1, match_func = lambda data, result: _masked_match( data, result, HV_CU_VAL1_TYPE, HV_NUM_CONDUNIT ) )
    ti.add_test_val( HV_CU_VAL1_LIKELY, sg.hv_val1_like_s, "HAVING_VAL1_LIKE", cond = lambda data, result: data[ PF_HAVING ] == 1 and 0 in data[ HV_CU_VAL1_TYPE ], match_func = lambda data, result: _masked_match( data, result, HV_CU_VAL1_LIKELY, HV_NUM_CONDUNIT, [ HV_CU_VAL1_TYPE ], [ 0 ] ) )
    ti.add_test_val( HV_CU_VAL1_BOOLVAL, sg.hv_val1_bool_s, "HAVING_VAL1_BOOL", cond = lambda data, result: data[ PF_HAVING ] == 1 and 1 in data[ HV_CU_VAL1_TYPE ], match_func = lambda data, result: _masked_match( data, result, HV_CU_VAL1_BOOLVAL, HV_NUM_CONDUNIT, [ HV_CU_VAL1_TYPE ], [ 1 ] ) )
    ti.add_test_val( HV_CU_VAL1_SP, sg.hv_val1_sp_s, "HAVING_VAL1_SP", cond = lambda data, result: data[ PF_HAVING ] == 1 and 0 in data[ HV_CU_VAL1_TYPE ], match_func = lambda data, result: _masked_match( data, result, HV_CU_VAL1_SP, HV_NUM_CONDUNIT, [ HV_CU_VAL1_TYPE ], [ 0 ] ) )
    ti.add_test_val( HV_CU_VAL1_EP, sg.hv_val1_ep_s, "HAVING_VAL1_EP", cond = lambda data, result: data[ PF_HAVING ] == 1 and 0 in data[ HV_CU_VAL1_TYPE ], match_func = lambda data, result: _masked_match( data, result, HV_CU_VAL1_EP, HV_NUM_CONDUNIT, [ HV_CU_VAL1_TYPE ], [ 0 ] ) )
    
    ti.add_test_val( HV_CU_VAL2_TYPE, sg.hv_val2_type_s, "HAVING_VAL2_TYPE", cond = lambda data, result: data[ PF_HAVING ] == 1 and 0 in data[ HV_CU_COND_OP ], match_func = lambda data, result: _masked_match( data, result, HV_CU_VAL2_TYPE, HV_NUM_CONDUNIT, [ HV_CU_COND_OP ], [ 0 ] ) )
    ti.add_test_val( HV_CU_VAL2_LIKELY, sg.hv_val2_like_s, "HAVING_VAL2_LIKE", cond = lambda data, result: data[ PF_HAVING ] == 1 and 0 in data[ HV_CU_COND_OP ] and 0 in data[ HV_CU_VAL2_TYPE ], match_func = lambda data, result: _masked_match( data, result, HV_CU_VAL2_LIKELY, HV_NUM_CONDUNIT, [ HV_CU_COND_OP, HV_CU_VAL2_TYPE ], [ 0, 0 ] ) )
    ti.add_test_val( HV_CU_VAL2_BOOLVAL, sg.hv_val2_bool_s, "HAVING_VAL2_BOOL", cond = lambda data, result: data[ PF_HAVING ] == 1 and 0 in data[ HV_CU_COND_OP ] and 0 in data[ HV_CU_VAL2_TYPE ], match_func = lambda data, result: _masked_match( data, result, HV_CU_VAL2_BOOLVAL, HV_NUM_CONDUNIT, [ HV_CU_COND_OP, HV_CU_VAL2_TYPE ], [ 0, 1 ] ) )
    ti.add_test_val( HV_CU_VAL2_SP, sg.hv_val2_sp_s, "HAVING_VAL2_SP", cond = lambda data, result: data[ PF_HAVING ] == 1 and 0 in data[ HV_CU_COND_OP ] and 0 in data[ HV_CU_VAL2_TYPE ], match_func = lambda data, result: _masked_match( data, result, HV_CU_VAL2_SP, HV_NUM_CONDUNIT, [ HV_CU_COND_OP, HV_CU_VAL2_TYPE ], [ 0, 0 ] ) )
    ti.add_test_val( HV_CU_VAL2_EP, sg.hv_val2_ep_s, "HAVING_VAL2_EP", cond = lambda data, result: data[ PF_HAVING ] == 1 and 0 in data[ HV_CU_COND_OP ] and 0 in data[ HV_CU_VAL2_TYPE ], match_func = lambda data, result: _masked_match( data, result, HV_CU_VAL2_EP, HV_NUM_CONDUNIT, [ HV_CU_COND_OP, HV_CU_VAL2_TYPE ], [ 0, 0 ] ) )

    ti.add_test_val( HV_CU_VU_OPERATOR, sg.hv_vu_op_s, "HAVING_VU_OP", cond = lambda data, result: data[ PF_HAVING ] == 1 )
    ti.add_test_val( HV_CU_VU_AGG1, sg.hv_vu_agg1_s, "HAVING_VU_AGG1", cond = lambda data, result: data[ PF_HAVING ] == 1 )
    ti.add_test_val( HV_CU_VU_COL1, sg.hv_vu_col1_s, "HAVING_VU_COL1", cond = lambda data, result: data[ PF_HAVING ] == 1 )
    ti.add_test_val( HV_CU_VU_DIST1, sg.hv_vu_dist1_s, "HAVING_VU_DIST1", cond = lambda data, result: data[ PF_HAVING ] == 1 )
    ti.add_test_val( HV_CU_VU_AGG2, sg.hv_vu_agg2_s, "HAVING_VU_AGG2", cond = lambda data, result: data[ PF_HAVING ] == 1 and max( data[ HV_CU_VU_OPERATOR ] ) > 0, match_func = lambda data, result: _arg2_test( data, result, HV_CU_VU_AGG2, HV_CU_VU_OPERATOR, HV_NUM_CONDUNIT ) )
    ti.add_test_val( HV_CU_VU_COL2, sg.hv_vu_col2_s, "HAVING_VU_COL2", cond = lambda data, result: data[ PF_HAVING ] == 1 and max( data[ HV_CU_VU_OPERATOR ] ) > 0, match_func = lambda data, result: _arg2_test( data, result, HV_CU_VU_COL2, HV_CU_VU_OPERATOR, HV_NUM_CONDUNIT ) )
    ti.add_test_val( HV_CU_VU_DIST2, sg.hv_vu_dist2_s, "HAVING_VU_DIST2", cond = lambda data, result: data[ PF_HAVING ] == 1 and max( data[ HV_CU_VU_OPERATOR ] ) > 0, match_func = lambda data, result: _arg2_test( data, result, HV_CU_VU_DIST2, HV_CU_VU_OPERATOR, HV_NUM_CONDUNIT ) )

def add_tbl_test_info( ti, sg ):
    ti.add_test_val( TV_TABLES_NUM, sg.tbl_num_r, "TABLE NUM", is_tbl = True, match_func = lambda data, result: data[ TV_TABLES_NUM ] == result[ TV_TABLES_NUM ] )
    ti.add_test_val( TV_TABLES_USED_IDX, sg.tables_s, "USED TABLES", match_func = lambda data, result: set( data[ TV_TABLES_USED_IDX ] ) == set( np.array( result[ TV_TABLES_USED_IDX ] ).argsort()[-result[ TV_TABLES_NUM ]: ].tolist() ), is_tbl = True )


