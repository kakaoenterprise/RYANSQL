import json


import sys

from dbinfo import *
from qinfo  import *

class DataStorage:
    def __init__( self ):
        self.map_dbs    = dict()    # DB ID - Database info.

    def load_db_from_file( self, db_fn ):
        db_fin      = open( db_fn, "r" )
        vec_db_info = json.load( db_fin )
        for db_info in vec_db_info:
            db  = DBInfo( db_info )
            if db.db_id in self.map_dbs:
                print ( "DUPLICATED DATABASE ID: [%s]" % db.db_id )
                sys.exit(0)
            self.map_dbs[ db.db_id ]    = db
        db_fin.close()
        print ( "TOTAL %s databases are read." % len( self.map_dbs ) )

    def load_train_datasets( self, vec_train_fn, valid_fn ):
        self.vec_train  = []
        for fn in vec_train_fn:
            self.vec_train  += self._load_qdata_from_file( fn )

        self.vec_valid  = self._load_qdata_from_file( valid_fn )

        print ( "TRAIN DATA NUM: [%d]" % len( self.vec_train ) )
        print ( "VALID DATA NUM: [%d]" % len( self.vec_valid ) )
    
    def load_valid_datasets( self, valid_fn ):
        self.vec_valid  = self._load_qdata_from_file( valid_fn )

        print ( "VALID DATA NUM: [%d]" % len( self.vec_valid ) )

    def get_prop_cla_features( self, use_se = False, merged_col_name = False, cn_type = 0 ):
        if merged_col_name:
            cn_type = 1

        vec_ret_train   = []
        vec_ret_valid   = []
        for qinfo in self.vec_train:
            vec_ret_train   += qinfo.get_prop_cla_features( use_se = use_se, cn_type = cn_type )

        for qinfo in self.vec_valid:
            vec_ret_valid   += qinfo.get_prop_cla_features( use_se = use_se, cn_type = cn_type )

        return vec_ret_train, vec_ret_valid

    def get_actual_input_features( self, merged_col_name = False, cn_type = 0 ):
        if merged_col_name:
            cn_type = 1

        vec_ret = []
        for qinfo in self.vec_valid:
            vec_ret.append( qinfo.get_actual_input_features( cn_type = cn_type ) )

        return vec_ret

    # Returns: list of QInfo.
    def _load_qdata_from_file( self, q_fn ):
        vec_ret     = []

        q_fin       = open( q_fn, "r" )
        vec_q_info  = json.load( q_fin )
        for json_q_info in vec_q_info:
            q_info  = QInfo( json_q_info[ "question" ], json_q_info[ "question_toks" ], json_q_info[ "query" ], json_q_info[ "db_id" ] )
            db      = self.map_dbs[ q_info.db_id ]

            sql_query   = json_q_info[ "sql" ]
            if sql_query[ "from" ][ "table_units" ][ 0 ][0] == "sql":
                sql_query           = sql_query[ "from" ][ "table_units" ][0][1]
                q_info._b_from_sql  = True

            q_info.initialize_train_qinfo( sql_query, db )
            vec_ret.append( q_info )

        q_fin.close()

        return vec_ret

if __name__ == "__main__":
    ds  = DataStorage()
    ds.load_db_from_file( "../dataset/tables.json" )
    ds.load_train_datasets( [ "../dataset/train_spider.json", "../dataset/train_others.json" ], "../dataset/dev.json" )
    vec_features, vf_2     = ds.get_prop_cla_features()
    set_types   = set()
    for v_f in vec_features:
        for c_t in v_f[ PF_TCOLTYPE ]:
            for t in c_t:
                set_types.add( t )
    print ( len( set_types ) )
    for t in set_types:
        print  (t )
#    for v in vec_features:
#        print ( v[ WF_CU_VAL1_TYPE ] )
   #     print ( v[ PF_WHERE ] )
  #      print ( "------" )
    numcols = [ len( v[ TV_TABLES_USED_IDX ] ) for v in vec_features + vf_2 ]
    print ( max( numcols ) ) 

######TYPES
'''
time
number
text
others
boolean
'''
