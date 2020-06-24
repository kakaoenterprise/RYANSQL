import sys
sys.path.append( "../../util" )
sys.path.append( "../../util/bert" )


from db_meta    import *

def get_used_glove( vec_prop_f, set_out, target_dic ):
    for v_f in vec_prop_f:
        for t in v_f[ PF_QTOKS ]:
            if t in target_dic:
                set_out.add( t )
            elif t.lower() in target_dic:
                set_out.add( t.lower() )

        for vec_table_name in v_f[ TV_TABLES_NAME ]:
            for t in vec_table_name:
                if t in target_dic:
                    set_out.add( t )
                elif t.lower() in target_dic:
                    set_out.add( t.lower() )

        for v_tbl in v_f[ PF_TCOLTOKS ]:
            for v_c in v_tbl:
                for t in v_c:
                    if t in target_dic:
                        set_out.add( t )
                    elif t.lower() in target_dic:
                        set_out.add( t.lower() )

def get_char_dic( vec_prop_f, cur_dic ):
    for v_f in vec_prop_f:
        for t in v_f[ PF_QTOKS ]:
            for c in t:
                if c not in cur_dic:
                    cur_dic[c]  = len( cur_dic )

        for vec_table_name in v_f[ TV_TABLES_NAME ]:
            for t in vec_table_name:
                for c in t:
                    if c not in cur_dic:
                        cur_dic[c]  = len( cur_dic )

        for v_tbl in v_f[ PF_TCOLTOKS ]:
            for v_c in v_tbl:
                for t in v_c:
                    for c in t:
                        if c not in cur_dic:
                            cur_dic[c]  = len( cur_dic )
    return cur_dic

def _get_dic_id( t, dic ):
    if t in dic:
        return dic[t]
    elif t.lower() in dic:
        return dic[ t.lower() ]
    return len( dic )

# Attach GloVe IDs to the property extraction features.
def get_glove_ids( vec_prop_f, dic ):
    for vec_f in vec_prop_f:
        vec_f[ PF_QGLOVE ]  = []
        for t in vec_f[ PF_QTOKS ]:
            vec_f[ PF_QGLOVE ].append( _get_dic_id( t, dic ) )

        vec_f[ TV_TABLES_NAME_GLOVE ]   = []
        for table_name in vec_f[ TV_TABLES_NAME ]:
            vec_tbl_glove   = []
            for t in table_name:
                vec_tbl_glove.append( _get_dic_id( t, dic ) )
            vec_f[ TV_TABLES_NAME_GLOVE ].append( vec_tbl_glove )

        vec_f[ PF_TCOLGLOVE ]   = []
        for vec_tbl in vec_f[ PF_TCOLTOKS ]:
            vec_tbl_glove   = []
            for vec_c in vec_tbl:
                vec_col_glove   = []
                for ct in vec_c:
                    vec_col_glove.append( _get_dic_id( ct, dic ) )
                vec_tbl_glove.append( vec_col_glove )
            vec_f[ PF_TCOLGLOVE ].append( vec_tbl_glove )
        
        vec_f[ PF_TCOLTBLGLOVE ]   = []
        for vec_tbl in vec_f[ PF_TCOLTBLN ]:
            vec_tbl_glove   = []
            for vec_c in vec_tbl:
                vec_col_glove   = []
                for ct in vec_c:
                    vec_col_glove.append( _get_dic_id( ct, dic ) )
                vec_tbl_glove.append( vec_col_glove )
            vec_f[ PF_TCOLTBLGLOVE ].append( vec_tbl_glove )

# Attach CHAR IDs to the property extraction features.
def get_char_ids( vec_prop_f, dic ):
    for vec_f in vec_prop_f:
        vec_f[ PF_QCHAR ]  = []
        for t in vec_f[ PF_QTOKS ]:
            vec_token_char_idx  = []
            for char in t:
                vec_token_char_idx.append( _get_dic_id( char, dic ) )
            vec_f[ PF_QCHAR ].append( vec_token_char_idx )
        
        vec_f[ TV_TABLES_NAME_CHAR ]   = []
        for table_name in vec_f[ TV_TABLES_NAME ]:
            vec_tbl_token  = []
            for t in table_name:
                vec_tbl_token_char  = []
                for char in t:
                    vec_tbl_token_char.append( _get_dic_id( char, dic ) )
                vec_tbl_token.append( vec_tbl_token_char )                
            vec_f[ TV_TABLES_NAME_CHAR ].append( vec_tbl_token )

        vec_f[ PF_TCOLCHAR ]   = []
        for vec_tbl in vec_f[ PF_TCOLTOKS ]:
            vec_tbl_token   = []
            for vec_c in vec_tbl:
                vec_col_token   = []
                for ct in vec_c:
                    vec_col_char    = []
                    for char in ct:
                        vec_col_char.append( _get_dic_id( char, dic ) )
                    vec_col_token.append( vec_col_char )
                vec_tbl_token.append( vec_col_token )
            vec_f[ PF_TCOLCHAR ].append( vec_tbl_token )
        
        vec_f[ PF_TCOLTBLCHAR ]   = []
        for vec_tbl in vec_f[ PF_TCOLTBLN ]:
            vec_tbl_token   = []
            for vec_c in vec_tbl:
                vec_col_token   = []
                for ct in vec_c:
                    vec_col_char    = []
                    for char in ct:
                        vec_col_char.append( _get_dic_id( char, dic ) )
                    vec_col_token.append( vec_col_char )
                vec_tbl_token.append( vec_col_token )
            vec_f[ PF_TCOLTBLCHAR ].append( vec_tbl_token )

# Generate BERT Tokens.
# Convert to ID during the feature generation procedure.
# Q_BERT_TOK: Q.
# C_BERT_TOK: T X C.
def get_bert_tokens( vec_prop_f, bert_tokenizer ):
    vec_Q_pointers      = [ LF_POINTERLOC, WF_CU_VAL1_SP, WF_CU_VAL2_SP, HV_CU_VAL1_SP, HV_CU_VAL2_SP ]
    vec_Q_pointers_E    = [ WF_CU_VAL1_EP, WF_CU_VAL2_EP, HV_CU_VAL1_EP, HV_CU_VAL2_EP ]
    for vec_f in vec_prop_f:
        map_glove_bert_idx  = dict()
        q_bert_loc          = 0
        vec_f[ Q_BERT_TOK ] = []
        for tidx, t in enumerate( vec_f[ PF_QTOKS ] ):
            t_tokens    = bert_tokenizer.tokenize( t )
            vec_f[ Q_BERT_TOK ]         += t_tokens
            map_glove_bert_idx[ tidx ]  = q_bert_loc 
            q_bert_loc                  += len( t_tokens )

        vec_f[ C_BERT_TOK ] = []    # T X C X col_tokens.
        for vec_tbl in vec_f[ PF_TCOLTOKS ]:
            vec_tbl_bert    = []
            for vec_c in vec_tbl:
                vec_c_bert  = []
                for ct in vec_c:
                    vec_c_bert  += bert_tokenizer.tokenize( ct )
                vec_tbl_bert.append( vec_c_bert )
            vec_f[ C_BERT_TOK ].append( vec_tbl_bert )
        
        # Do Location Mapping.
        if LF_POINTERLOC in vec_f:
            vec_f[ LF_POINTERLOC ]  = [ vec_f[ LF_POINTERLOC ] ]
            for pointer_prop in vec_Q_pointers:
                vec_f[ pointer_prop ]   = [ map_glove_bert_idx[ v ] for v in vec_f[ pointer_prop ] ]       
            vec_f[ LF_POINTERLOC ]  = vec_f[ LF_POINTERLOC ][0]
            for pointer_prop in vec_Q_pointers_E:
                vec_f[ pointer_prop ]   = [ map_glove_bert_idx[ v + 1 ] - 1 if v + 1 in map_glove_bert_idx else len( vec_f[ Q_BERT_TOK ] ) - 1 for v in vec_f[ pointer_prop ] ]



