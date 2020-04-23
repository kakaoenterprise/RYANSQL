PATH_NONE   = "NONE"
PATH_UNION  = "UNION"
PATH_INTER  = "INTERSECT"
PATH_EXCEPT = "EXCEPT"
PATH_WHERE  = "WHERE"
PATH_HAVING = "HAVING"
PATH_PAR    = "PARALLEL"    # To represent the multiple selection clauses in a single WHERE clause.

VEC_AGGREGATORS = [ 'none', 'max', 'min', 'count', 'sum', 'avg' ]
VEC_OPERATORS   = [ 'none', '-', '+', "*", '/' ]
VEC_CONDOPS     = [ 'between', '=', '>', '<', '>=', '<=', '!=', 'in', 'like', 'is', 'exists' ]
IDX_DESCRIPTORS = { 'asc': 0, 'desc': 1 }
IDX_INV_DESCRIPTORS  = { v:k for k,v in IDX_DESCRIPTORS.items() }

MERGE_NONE      = "NONE"
MERGE_UNION     = "UNION"
MERGE_INTER     = "INTERSECT"
MERGE_EXCEPT    = "EXCEPT"
IDX_MERGE_OP    = { MERGE_NONE: 0, MERGE_UNION: 1, MERGE_INTER: 2, MERGE_EXCEPT: 3 }
IDX_PATH        = { PATH_NONE: 0, PATH_UNION: 1, PATH_INTER: 2, PATH_EXCEPT: 3, PATH_WHERE: 4, PATH_HAVING: 5, PATH_PAR: 6 }

# DB PROP CLASSIFICATION FEATURES.
PF_PATH     = "cur_path"
PF_PATHIDX  = "cur_path_idx"
PF_MERGEOP  = "merge_op"    # Valid only when path == [ NONE ]
PF_FROMSQL  = "sql_from"    # Valid only when path == [ NONE ]
PF_ORDERBY  = "is_orderby"
PF_GROUPBY  = "is_groupby"
PF_LIMIT    = "has_limit"
PF_WHERE    = "has_where"
PF_HAVING   = "has_having"
PF_QTOKS    = "question_tokens"
PF_QGLOVE   = "question_glove"
PF_QCHAR    = "question_char_idx"
PF_QMATCH   = "question_to_table_match_info" 

PF_TCOLTOKS = "table_column_tokens"
PF_TCOLTBLN = "table_column_tbl_name" 
PF_TCOLTYPE = "table_column_types"
PF_TCOLTYPEID   = "table_column_type_ids"
PF_TCOLGLOVE    = "table_column_glove"
PF_TCOLTBLGLOVE    = "table_column_table_glove"
PF_TMATCH       = "table_to_question_match_info" 
PF_TPHMATCH     = "table_column_exists_in_question_as_a_whole"      # NOT IMPLEMENTED YET.
PF_TKEYPRIMARY  = "table_column_is_primary"
PF_TKEYFOREIGN  = "table_column_is_foreign"
PF_Q        = "question"
PF_SQL      = "sql" 

PF_TCOLTOKS_RAW = "table_column_tokens_raw"
PF_TCOLTOKS_RAW_GLOVE = "table_column_tokens_raw_glove"
PF_QMATCH_RAW   = "question_to_table_match_info_raw"
PF_TMATCH_RAW   = "table_to_question_match_info_raw"
PF_TCOLCHAR_RAW = "table_column_char_idx_raw"

PF_TCOLCHAR     = "table_column_char_idx"
PF_TCOLTBLCHAR  = "table_column_table_char_idx"

# GROUP_BY CLASSIFICATION FIELDS.
GF_NUMCOL   = "groupby_col_num"
GF_COLLIST  = "groupby_col_list" 

# ORDER_BY CLASSSIFICATION FIELDS.
OF_DESCRIPTOR   = "orderby_descriptor"
OF_NUMVU        = "orderby_num_valueunit" 
OF_VU_OPERATOR  = "orderby_valueunit_operator"
OF_VU_AGG1      = "orderby_valueunit_agg1" 
OF_VU_COL1      = "orderby_valueunit_col1" 
OF_VU_DIST1     = "orderby_valueunit_isdist1"
OF_VU_AGG2      = "orderby_valueunit_agg2"
OF_VU_COL2      = "orderby_valueunit_col2"
OF_VU_DIST2     = "orderby_valueunit_isdist2" 

# SELECT CLASSIFICATION FIELDS.
SF_DISTINCT     = "select_distinct"
SF_NUM_VU       = "select_num_valueunit"
SF_VU_OPERATOR  = "select_valueunit_operator"
SF_VU_AGG1      = "select_valueunit_agg1" 
SF_VU_COL1      = "select_valueunit_col1" 
SF_VU_DIST1     = "select_valueunit_isdist1"
SF_VU_AGG2      = "select_valueunit_agg2"
SF_VU_COL2      = "select_valueunit_col2"
SF_VU_DIST2     = "select_valueunit_isdist2" 
SF_VU_AGGALL    = "select_valueunit_aggregator"

# LIMIT CLASSIFICATION FIELDS.
LF_ISMAX        = "limit_ismax"
LF_POINTERLOC   = "limit_pointer_loc"   # Valid only when LF_ISMAX is False.

# WHERE CLASSIFICATION FIELDS.
WF_NUM_CONDUNIT     = "where_num_condunit"
WF_CU_AGGREGATOR    = "where_cu_aggregator"
WF_CU_IS_NOT        = "where_cu_is_not"
WF_CU_COND_OP       = "where_cu_condop"
WF_CU_VAL1_TYPE     = "where_cu_val1_type"          # 0: Text span, 1: BOOLEAN, 2: SELECT statement.
WF_CU_VAL1_SP       = "where_cu_val1_sp"            # Valid only when WF_CU_VAL1_TYPE is text span.
WF_CU_VAL1_EP       = "where_cu_val1_ep"            # Valid only when WF_CU_VAL1_TYPE is text span.
WF_CU_VAL1_LIKELY   = "where_cu_val1_likely"        # 0: Exact match. 1: Front-likely. 2: Backward-likely. 3: Both side likely.
WF_CU_VAL1_BOOLVAL  = "where_cu_val1_boolean"       # If the condition is expected to be true or false.
WF_CU_VAL2_TYPE     = "where_cu_val2_type"          # 0: Text span, 1: BOOLEAN.
WF_CU_VAL2_SP       = "where_cu_val2_sp"            # Valid only when WF_CU_COND_OP == 0 (between)
WF_CU_VAL2_EP       = "where_cu_val2_ep"            # Valid only when WF_CU_COND_OP == 0 (between)
WF_CU_VAL2_LIKELY   = "where_cu_val2_likely"        # 0: Exact match. 1: Front-likely. 2: Backward-likely. 3: Both side likely.
WF_CU_VAL2_BOOLVAL  = "where_cu_val2_boolean"       # If the condition is expected to be true or false.
WF_CU_VU_OPERATOR   = "where_cu_valueunit_operator"
WF_CU_VU_AGG1       = "where_cu_valueunit_agg1" 
WF_CU_VU_COL1       = "where_cu_valueunit_col1" 
WF_CU_VU_DIST1      = "where_cu_valueunit_isdist1"
WF_CU_VU_AGG2       = "where_cu_valueunit_agg2"
WF_CU_VU_COL2       = "where_cu_valueunit_col2"
WF_CU_VU_DIST2      = "where_cu_valueunit_isdist2" 

WF_CU_VAL1_IGNORE   = "where_ignore_val1"           # Answer matching failed.
WF_CU_VAL2_IGNORE   = "where_ignore_val2"           # Answer matching failed.

# HAVING CLASSIFICATION FIELDS.
HV_NUM_CONDUNIT     = "having_num_condunit"
HV_CU_AGGREGATOR    = "having_cu_aggregator"
HV_CU_IS_NOT        = "having_cu_is_not"
HV_CU_COND_OP       = "having_cu_condop"
HV_CU_VAL1_TYPE     = "having_cu_val1_type"          # 0: Text span, 1: BOOLEAN, 2: SELECT statement.
HV_CU_VAL1_SP       = "having_cu_val1_sp"            # Valid only when HV_CU_VAL1_TYPE is text span.
HV_CU_VAL1_EP       = "having_cu_val1_ep"            # Valid only when HV_CU_VAL1_TYPE is text span.
HV_CU_VAL1_LIKELY   = "having_cu_val1_likely"        # 0: Exact match. 1: Front-likely. 2: Backward-likely. 3: Both side likely.
HV_CU_VAL1_BOOLVAL  = "having_cu_val1_boolean"       # If the condition is expected to be true or false.
HV_CU_VAL2_TYPE     = "having_cu_val2_type"          # 0: Text span, 1: BOOLEAN.
HV_CU_VAL2_SP       = "having_cu_val2_sp"            # Valid only when HV_CU_COND_OP == 0 (between)
HV_CU_VAL2_EP       = "having_cu_val2_ep"            # Valid only when HV_CU_COND_OP == 0 (between)
HV_CU_VAL2_LIKELY   = "having_cu_val2_likely"        # 0: Exact match. 1: Front-likely. 2: Backward-likely. 3: Both side likely.
HV_CU_VAL2_BOOLVAL  = "having_cu_val2_boolean"       # If the condition is expected to be true or false.
HV_CU_VU_OPERATOR   = "having_cu_valueunit_operator"
HV_CU_VU_AGG1       = "having_cu_valueunit_agg1" 
HV_CU_VU_COL1       = "having_cu_valueunit_col1" 
HV_CU_VU_DIST1      = "having_cu_valueunit_isdist1"
HV_CU_VU_AGG2       = "having_cu_valueunit_agg2"
HV_CU_VU_COL2       = "having_cu_valueunit_col2"
HV_CU_VU_DIST2      = "having_cu_valueunit_isdist2" 

HV_CU_VAL1_IGNORE   = "having_ignore_val1"           # Answer matching failed.
HV_CU_VAL2_IGNORE   = "having_ignore_val2"           # Answer matching failed.

# TABLE CLASSIFICATION FEATURES.
TV_TABLES_NAME          = "table_name"
TV_TABLES_NAME_GLOVE    = "table_name_glove_idx"
TV_TABLES_NAME_CHAR     = "table_name_char_idx"
TV_NAME_EXACT           = "table_name_exact_match"

# TABLE CLASSIFICATION CLASSES.
TV_TABLES_NUM       = "table_used_num" 
TV_TABLES_USED_IDX  = "tables_used_idx"

# DB META.
META_DB     = "db"

# BERT_RELATED.
Q_BERT_TOK  = "q_bert_tok"  # Question BERT Tokens.
C_BERT_TOK  = "c_bert_tok"  # Column BERT Tokens.
BERT_ID     = "bert_id"     # BERT IDs.

