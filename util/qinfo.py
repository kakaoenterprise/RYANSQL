from select_query   import *
from nltk.stem      import PorterStemmer
porter_stemmer  = PorterStemmer()

class QInfo:
    def __init__( self, question, vec_q_toks, query_str, db_id ):
        self.question       = question
        self.vec_q_toks     = vec_q_toks
        self.query_str      = query_str
        self.db_id          = db_id

        self._merge_op      = MERGE_NONE
        self._main_select   = None
        self._sub_select    = None  # Valid only when merge_op is not None.
        self._b_from_sql    = False
      
    def initialize_train_qinfo( self, q_json, db ):
        self.db             = db
        self._main_select   = SelectQuery()
        self._main_select.initialize_query_from_json( q_json, self.vec_q_toks, db )
    
        sub_select_info = None
        if q_json[ "union" ] != None:
            self._merge_op  = MERGE_UNION
            sub_select_info = q_json[ "union" ]
        elif q_json[ "intersect" ] != None:
            self._merge_op  = MERGE_INTER
            sub_select_info = q_json[ "intersect" ]
        elif q_json[ "except" ] != None:
            self._merge_op  = MERGE_EXCEPT
            sub_select_info = q_json[ "except" ]

        if sub_select_info != None:
            self._sub_select    = SelectQuery()
            self._sub_select.initialize_query_from_json( sub_select_info, self.vec_q_toks, db )

    def get_query( self ):
        query_str   = ""
        if self._merge_op != "none":
            query_str   = "%s %s %s" % ( self._main_select.get_query(), self._merge_op, self._sub_select.get_query() )
        else:
            query_str   = self._main_select.get_query() 

        if self._b_from_sql:
            query_str   = "SELECT COUNT(*) FROM ( %s )" % query_str

        return query_str

    # Gets Features for property classification.
    # Generate all instances which could be gathered from this question.
    # Returns: Vectors of Training Data dictionaries.
    def get_prop_cla_features( self, use_se = False, merged_col_name = False, cn_type = 0 ):
        if merged_col_name:
            cn_type = 1

        # 1. Get all the selection query property infos
        vec_features        = []
        vec_main_features   = self._main_select.get_prop_cla_features()
        for vec_f in vec_main_features:
            if len( vec_f[ PF_PATH ] ) == 0:
                vec_f[ PF_PATH ]    = [ PATH_NONE ] + vec_f[ PF_PATH ]
            vec_features.append( vec_f )

        if self._sub_select != None:
            vec_sub_features    = self._sub_select.get_prop_cla_features()
            sub_path            = ""
            if self._merge_op == MERGE_UNION:
                sub_path    = PATH_UNION
            elif self._merge_op == MERGE_INTER:
                sub_path    = PATH_INTER
            elif self._merge_op == MERGE_EXCEPT:
                sub_path    = PATH_EXCEPT
            else:
                print ( "ERROR IN SUBPATH: NO SUBPATH TYPE!" )
                return []

            for vec_f in vec_sub_features:
                vec_f[ PF_PATH ]    = [ sub_path ] + vec_f[ PF_PATH ]
                vec_features.append( vec_f )

        f_general   = self.get_actual_input_features( cn_type = cn_type )
        
        # Add tokens & database columns.
        for vec_f in vec_features:
            vec_f[ PF_FROMSQL ]    = int( self._b_from_sql )
            vec_f[ PF_MERGEOP ]     = IDX_MERGE_OP[ self._merge_op ] 

            for k, v in f_general.items():
                vec_f[ k ]  = v

        return vec_features

    # Return the data in actual input manner.
    def get_actual_input_features( self, merged_col_name = False, cn_type = 0 ):
        if merged_col_name:
            cn_type = 1 # Backward compatibility.

        t_col_tok_tbl           = [ [ c.get_rep_name( cn_type = cn_type ) for c in t.vec_cols ] for t in self.db.vec_tbls ]         # Table-specific one. without the special "*" column.

        t_col_tbl_name          = [ [ t.get_rep_name() for c in t.vec_cols ] for t in self.db.vec_tbls ]
        t_col_type_tbl          = [ [ c.col_type for c in t.vec_cols ] for t in self.db.vec_tbls ]
        t_key_primary_tbl       = [ [ c.is_primary for c in t.vec_cols ] for t in self.db.vec_tbls ]
        t_key_foreign_tbl       = [ [ c.is_foreign for c in t.vec_cols ] for t in self.db.vec_tbls ]

        vec_qtok_stemmed        = [ porter_stemmer.stem( t.lower() ) for t in self.vec_q_toks ]
        set_qtok_stemmed        = set( vec_qtok_stemmed )
        set_ttok_stemmed        = set( [ porter_stemmer.stem( t.lower() ) for table in t_col_tok_tbl for col in table for t in col ] )

        qtok_match      = [ porter_stemmer.stem( t.lower() ) in set_ttok_stemmed and t != "," and t != "." for t in self.vec_q_toks ]
        ttok_match_tbl  = [ [ [ porter_stemmer.stem( t.lower() ) in set_qtok_stemmed and t != "," and t != "." for t in col ] for col in table ] for table in t_col_tok_tbl ]       # B X C X T.
        
        # Raw tbl col info without tbl marks.
        t_col_tok_raw           = [ [ c.get_rep_name( cn_type = 0 ) for c in t.vec_cols ] for t in self.db.vec_tbls ]
        set_ttok_raw_stemmed    = set( [ porter_stemmer.stem( t.lower() ) for table in t_col_tok_raw for col in table for t in col ] )
        qtok_match_raw          = [ porter_stemmer.stem( t.lower() ) in set_ttok_raw_stemmed and t != "," and t != "." for t in self.vec_q_toks ]
        ttok_match_raw          =  [ [ [ porter_stemmer.stem( t.lower() ) in set_qtok_stemmed and t != "," and t != "." for t in col ] for col in table ] for table in t_col_tok_raw ]
        
        q_tok           = [ t for t in self.vec_q_toks ]
        
        vec_f                   = dict()
        vec_f[ PF_QTOKS ]       = q_tok
        vec_f[ PF_QCHAR ]       = [ [ char for char in word.lower() ] for word in q_tok ]   
        vec_f[ PF_QMATCH ]      = qtok_match
        vec_f[ PF_QMATCH_RAW ]  = qtok_match_raw
        vec_f[ PF_TCOLTOKS ]    = t_col_tok_tbl     # B X C X T. B: # of tables.
        vec_f[ PF_TCOLTOKS_RAW ]    = t_col_tok_raw     # B X C X T. B: # of tables.
        vec_f[ PF_TCOLTBLN ]    = t_col_tbl_name    # B X C X T. B: # of Tables. Table name vector for each column,
        vec_f[ PF_TCOLTYPE ]    = t_col_type_tbl    # B X C
        vec_f[ PF_TCOLCHAR ]    = [ [ [ [ char for char in word.lower() ] for word in col ] for col in vec_cols ] for vec_cols in t_col_tok_tbl ]
        vec_f[ PF_TCOLCHAR_RAW ]    = [ [ [ [ char for char in word.lower() ] for word in col ] for col in vec_cols ] for vec_cols in t_col_tok_raw ]

        vec_f[ PF_TKEYPRIMARY ] = t_key_primary_tbl    # B X C.
        vec_f[ PF_TKEYFOREIGN ] = t_key_foreign_tbl    # B X C.
        vec_f[ PF_TMATCH ]      = ttok_match_tbl    # B X C X T
        vec_f[ PF_TMATCH_RAW ]  = ttok_match_raw    # B X C X T
        vec_f[ PF_Q ]           = self.question
        vec_f[ PF_SQL ]         = self.query_str
        vec_f[ TV_TABLES_NAME ]  = [ t.get_rep_name() for t in self.db.vec_tbls ]    # B X T'.
        vec_f[ TV_NAME_EXACT ]  = []        # B.
        for t in self.db.vec_tbls:
            vec_tname       = t.vec_norm_name
            match_flag      = False
            for qidx, qt in enumerate( vec_qtok_stemmed ):
                if qt == vec_tname[0] and qidx + len( vec_tname ) <= len( vec_qtok_stemmed ):
                    match_flag  = True
                    for compare_idx in range( len( vec_tname ) ):
                        if vec_qtok_stemmed[ qidx + compare_idx ] != vec_tname[ compare_idx ]:
                            match_flag  = False
                            break
                    if match_flag:
                        break
            vec_f[ TV_NAME_EXACT ].append( match_flag )

        vec_f[ META_DB ]        = self.db

        return vec_f
