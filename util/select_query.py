from db_meta    import *

def is_float( w ):
    try:
        float( w )
    except ValueError:
        return False

    return True

def is_int( w ):
    if is_float(w) and int( float( w ) ) == float( w ):
        return True

    try:
        int( w )
    except ValueError:
        return False

    return True

def match_word( w1, w2 ):
    map_num_str = { 0: ["zero"], 1:["one", "single", "once" ], 2:["two", "twice"], 3:["three"], 4:["four"], 5:["five"], 6:["six"], 7:["seven"], 8:["eight"], 9:["nine"], 10:["ten" ] }
    same_mean   = { 0: [ "France", "French" ], 1: [ "F", "female", "Female" ], 2: [ "M", "male", "Male" ], 3: [ "CA", "California" ], 4: [ "Africa", "African" ], 5: [ "Asia", "Asian" ], 6: [ "Brazil", "Brazil's" ], \
                    7: [ "engineer", "engineering" ], 8: [ "Europe", "europe", "European" ], 9: [ "UK", "British" ], 10: [ "Mortgages", "Mortage" ], 11: [ "8/", "August" ], 12: [ "12/", "December" ], \
                    13: [ "foggy", "Fog" ], 14: [ "rained", "Rain" ], 15: [ "Illinois", "IL" ], 16: [ "US", "USA" ], 17: [ "activator", "activitor" ], 18: [ "left", "left-footed" ], 19: [ "Uniform", "uniformed" ], \
                    20: [ "Brown", "Brown's" ], 21: [ "Annaual", "Annual" ], 22: [ "Success", "successful" ], 23: [ "Fail", "failure" ], 24: [ "PROF", "professors" ], 25: [ "Ph.D.", "Ph.D" ] }
    mean_dict   = dict()
    for k, vec in same_mean.items(): 
        for v in vec:
            mean_dict[ v ]  = k

    if w1 == w2 or w1.lower() == w2.lower():
        return True

    if w1[-1] == 's' and w1[:-1].lower() == w2.lower():
        return True

    if w2[-1] == 's' and w2[:-1].lower() == w1.lower():
        return True

    if w1[0] == "'" and w1[1:].lower() == w2.lower():
        return True
    
    if w2[0] == "'" and w2[1:].lower() == w1.lower():
        return True

    if w1[-1] == "'" and w1[:-1].lower() == w2.lower():
        return True
    
    if w2[-1] == "'" and w2[:-1].lower() == w1.lower():
        return True

    if w1[0] == "'" and w1[-1] == "'" and w1[1:-1].lower() == w2.lower():
        return True
    
    if w2[0] == "'" and w2[-1] == "'" and w2[1:-1].lower() == w1.lower():
        return True

    if is_float( w1 ) and is_float( w2 ) and float( w1 ) == float( w2 ):
        return True
   
    if is_int( w1 ) and int( float( w1 ) ) in map_num_str and w2.lower() in map_num_str[ int( float( w1 ) ) ]:
        return True
    
    if is_int( w2 ) and int( float( w2 ) ) in map_num_str and w1.lower() in map_num_str[ int( float( w2 ) ) ]:
        return True

    if w1 in mean_dict and w2 in mean_dict and mean_dict[ w1 ] == mean_dict[ w2 ]:
        return True
    
    if w1[0] == "‘" and w1[-1] == "’" and w1[1:-1].lower() == w2.lower():
        return True
    
    if w2[0] == "‘" and w2[-1] == "’" and w2[1:-1].lower() == w1.lower():
        return True

    return False

# Check to see if list_small is contained in list_large as its subsequence.
# If so, return start & end idx.
# Otherwise, return -1, -1.
def find_match_se( list_large, list_small ):
    for idx, t in enumerate( list_large ):
        if idx + len( list_small ) > len( list_large ):
            continue
        
        if match_word( t, list_small[0] ):
            flag    = True
            for sidx, t2 in enumerate( list_small ):
                if match_word( list_large[ idx + sidx ], t2 ):
                    continue
                flag = False

            if flag:
                return idx, idx + len( list_small ) - 1

    return -1, -1

# [ AGGERATOR, COLUMN, IS_DISTINCT ]
class Info_ColUnit:
    def __init__( self ):
        self.aggregator_idx = 0
        self.column         = None
        self.is_distinct    = False

    def read_from_json( self, col_json, db ):
        self.aggregator_idx = col_json[0]
        self.column         = db.vec_cols[ col_json[1] ]
        self.is_distinct    = col_json[2]

    def get_query( self ):
        q_str   = self.column.vec_cols[0].col_name_orig
        if self.aggregator_idx != 0:
            q_str   = "%s( %s )" % ( VEC_AGGREGATORS[ self.aggregator_idx ], q_str )

        if self.is_distinct:
            q_str   = "DISTINCT( %s )" % q_str

        return q_str
         

# [ OPERATOR, COLUNIT1, COLUNIT2 ]
class Info_ValueUnit:
    def __init__( self ):
        self.operator_idx   = 0
        self.col_left       = None
        self.col_right      = None
    
    def read_from_json( self, value_json, db ):
        self.operator_idx   = value_json[0]
        self.col_left       = Info_ColUnit()
        self.col_left.read_from_json( value_json[1], db )

        if self.operator_idx != 0:
            self.col_right  = Info_ColUnit()
            self.col_right.read_from_json( value_json[2], db )

    def get_query( self ):
        q_str   = self.col_left.get_query()
        if self.operator_idx != 0:
            q_str   = "%s %s %s" % ( q_str, VEC_OPERATORS[ self.operator_idx ], self.col_right.get_query() )

        return q_str          

# [ DESCRIPTOR, [ VALUEUNIT1, VALUEUNIT2,... ] ]
class Info_OrderBy:
    def __init__( self ):
        self.desc_idx       = 0
        self.vec_value_unit = []

    def read_from_json( self, orderby_json, db ):
        self.desc_idx       = IDX_DESCRIPTORS[ orderby_json[0] ]
        for vu_info in orderby_json[1]:
            vu  = Info_ValueUnit()
            vu.read_from_json( vu_info, db )
            self.vec_value_unit.append( vu )
    
    def get_query( self ):
        q_str   = ""
        for idx, vu in enumerate( self.vec_value_unit ):
            q_str   += vu.get_query()
            if idx < len( self.vec_value_unit ) - 1:
                q_str   += ", "
        q_str   = "%s %s" % ( q_str, IDX_INV_DESCRIPTORS[ self.desc_idx ].upper() )
        return q_str

# [ NOT_OP, COND_OP_ID, ValueUnit, val1, val2 ]
# Val2 is VALID ONLY WHEN COND_OP_ID is BETWEEN
# In that case, it's always INTEGER.
class Info_CondUnit:
    def __init__( self ):
        self.aggregator = 0     # and: 0 / or: 1
        self.is_not     = False
        self.cond_op_id = 1         # default: equal.
        self.value_unit = None
        self.val1       = None      # string / int / SelectQuery.
        self.val2       = None      # int. Available only when cond_op == BETWEEN.

    def has_select( self ):
        return isinstance( self.val1, SelectQuery )

    def read_from_json( self, cond_json, cond_agg, qtoks, db ):
        self.aggregator = 0 if cond_agg == "and" else 1
        self.is_not     = cond_json[0]
        self.cond_op_id = cond_json[1] - 1  # No "NOT"
        self.value_unit = Info_ValueUnit()
        self.value_unit.read_from_json( cond_json[2], db )
        self.val1_sp    = 0
        self.val1_ep    = 0
        self.val1_like  = 0 # 0: Exact / 1: Front-likely ( "%a" ) / 2: Backward-likely( "a%" ) / 3: Bothway likely.
        self.val1_qtype = 0 # 0: Text span. 1: Boolean. 2: Select.
        self.val1_boolval   = False
        self.val1_ignore    = False # Ignore the value 1, since we were not able to find the corresponding span.
        self.val2_sp        = 0
        self.val2_ep        = 0
        self.val2_like      = 0
        self.val2_isbool    = False
        self.val2_boolval   = False
        self.val2_ignore    = False
        self.val2_qtype     = 0
        if isinstance( cond_json[3], dict ):
            self.val1   = SelectQuery()
            self.val1.initialize_query_from_json( cond_json[3], qtoks, db )
            self.val1_qtype = 2
        else:
            self.val1   = str( cond_json[3] )

            # Find the location of val1.
            if self.val1[0] == '"' and self.val1[-1] == '"':
                self.val1   = self.val1[1:-1] 
            
            if len( self.val1 ) > 0 and self.val1[0] == '%' and self.val1[-1] == '%':
                self.val1_like  = 3
                self.val1   = self.val1[ 1:-1 ]
            elif len( self.val1 ) > 0 and self.val1[0] == '%':
                self.val1_like    = 1
                self.val1   = self.val1[ 1: ]
            elif len( self.val1 ) > 0 and self.val1[-1]  == '%':
                self.val1_like    = 2
                self.val1   = self.val1[ :-1 ]

            vec_val1   = [ t for t in str( self.val1 ).split( " " ) if t != "" ]
            if len( vec_val1 ) == 0:
                vec_val1.append( "empty" )

            s, e    = find_match_se( qtoks, vec_val1 )
            if s == -1: # Garbage collection.
                if self.val1 == "2.5" and "good" in qtoks:
                    idx = qtoks.index( "good" )
                    s   = idx
                    e   = idx
                if self.val1.lower() == "yes" or self.val1 == "T":
                    self.val1_qtype     = 1
                    self.val1_boolval   = True
                elif self.val1.lower() == "no" or self.val1 == "F":
                    self.val1_qtype     = 1

            if s == -1 and self.val1_qtype != 1:
                self.val1_ignore    = True
            else:
                if s != -1:
                    self.val1_sp    = s
                    self.val1_ep    = e

        if cond_json[4] != None:
            self.val2   = str( cond_json[4] )
            
            # Find the location of val1.
            if self.val2[0] == '"' and self.val2[-1] == '"':
                self.val2   = self.val2[1:-1] 
            
            if len( self.val2 ) > 0 and self.val2[0] == '%' and self.val2[-1] == '%':
                self.val2_like    = 3
                self.val2   = self.val2[ 1:-1 ]
            elif len( self.val2 ) > 0 and self.val2[0] == '%':
                self.val2_like    = 1
                self.val2   = self.val2[ 1: ]
            elif len( self.val2 ) > 0 and self.val2[-1]  == '%':
                self.val2_like    = 2
                self.val2   = self.val2[ :-1 ]

            vec_val2   = [ t for t in str( self.val2 ).split( " " ) if t != "" ]
            if len( vec_val2 ) == 0:
                vec_val2.append( "empty" )

            s, e    = find_match_se( qtoks, vec_val2 )
            if s == -1: # Garbage collection.
                if self.val2 == "2.5" and "good" in qtoks:
                    idx = qtoks.index( "good" )
                    s   = idx
                    e   = idx
                if self.val2.lower() == "yes" or self.val2 == "T":
                    self.val2_qtype     = 1
                    self.val2_boolval   = True
                elif self.val2.lower() == "no" or self.val2 == "F":
                    self.val2_qtype     = 1

            if s == -1 and not self.val2_qtype != 1:
                self.val2_ignore    = True
            else:
                if s != -1:
                    self.val2_sp    = s
                    self.val2_ep    = e

    def get_query( self ):
        q_str   = self.value_unit.get_query() + " "
        if self.is_not:
            q_str   += "NOT "
        q_str   += VEC_CONDOPS[ self.cond_op_id ] + " "
        if isinstance( self.val1, SelectQuery ):
            q_str   += "( %s )" % self.val1.get_query()
        else:
            q_str   += str( self.val1 )

        if self.val2 != None:
            q_str   += " AND %s" % self.val2

        return q_str


# [ CondUnit1, "and/or", CondUnit2, "and/or", ... ]
class Info_Condition:
    def __init__( self ):
        self.vec_condunits  = []

    def read_from_json( self, condition_json, qtoks, db ):
        cu  = Info_CondUnit()
        cu.read_from_json( condition_json[0], "and", qtoks, db )
        self.vec_condunits.append( cu )

        for idx in range( 2, len( condition_json ), 2 ):
            cu  = Info_CondUnit()
            cu.read_from_json( condition_json[ idx ], condition_json[ idx - 1 ], qtoks, db )
            self.vec_condunits.append( cu )

    def get_all_selections( self ):
        vec_ret = []
        for cu in self.vec_condunits:
            if cu.has_select():
                vec_ret.append( cu.val1 )
        return vec_ret

    def get_query( self ):
        q_str   = ""
        for idx, cu in enumerate( self.vec_condunits ):
            if idx > 0:
                q_str   += "%s " % cu.aggregator.upper()
            q_str   += cu.get_query() + " "
        
        return q_str.strip()

# [ IS_DISTINCT, [ [ AGG1, VALUEUNIT1 ], [ AGG2, VALUEUNIT2 ], ... ] ]
class Info_Select:
    def __init__( self ):
        self.is_distinct        = False
        self.vec_select_target  = []

    def read_from_json( self, select_json, db ):
        self.is_distinct    = select_json[0] 
        for agg_idx, vu_info in select_json[1]:
            vu  = Info_ValueUnit()
            vu.read_from_json( vu_info, db )
            self.vec_select_target.append( [ agg_idx, vu ] )

    def get_query( self ):
        q_str   = ""
        if self.is_distinct:
            q_str   = "DISTINCT "

        for aggidx, vu in self.vec_select_target:
            if aggidx == 0:
                q_str   += vu.get_query() + ", "
            else:
                q_str   += "%s( %s ), " % ( VEC_AGGREGATORS[ aggidx ], vu.get_query() )
    
        q_str   = q_str.strip()
        if q_str[-1] == ",":
            q_str   = q_str[:-1]

        return q_str

class SelectQuery:
    def __init__( self ):
        self._b_orderBy         = False
        self._b_groupBy         = False
        self._b_limit           = False
        self._b_where           = False
        self._b_having          = False

        # These 5 information are valid only when the boolean flag is True.
        self._info_orderBy      = None      # asc / desc + column list. NO NESTED ALLOWED.  - VALUE UNITS.  - IMPLEMENTED
        self._info_groupBy      = []        # Vector of Columns.                            CLASSIFY: # OF COLUMNS + TOP N COLUMNS. (Pointer Network)        - IMPLEMENTED.
        self._num_limit         = 0         # Always number.                                CLASSIFY: Pointer. ( TARGET: LIMIT )        - IMPLEMENTED
        self._limit_pos         = -1
        self._info_where        = None      # NESTED ALLOWED.   - IMPLEMENTED
        self._info_having       = None      # NESTED ALLOWED.

        # Select is ALWAYS required.
        self._info_select       = None      # CLASSIFY: Binary DISTINCT + # OF SELECTION N + N Aggregations + N Value Units.    - IMPLEMENTED

        # If the query is from SQL - final form will be wrapped with "SELECT count(*) FROM ...".
        # Otherwise, the FROM statements only require the set of required tables.
        self._info_from_tbl     = []        # CLASSIFY: # TABLES + TOP N TBL.

        # Metadata.
        self._db    = None
        
    def initialize_query_from_json( self, qjson, qtoks, db ):
        self._db    = db
        self._parse_orderBy( qjson, db )
        self._parse_groupBy( qjson, db )
        self._parse_limit( qjson, qtoks )
        self._parse_wherecond( qjson, qtoks, db )
        self._parse_having( qjson, qtoks, db )
        self._parse_select( qjson, db )
        self._parse_from( qjson, db )

    # Returns: Vector of Property classification Features.
    # Property Classification Features: A dictionary.
    def get_prop_cla_features( self ):
        vec_ret     = []

        # 1. Itself.
        features    = { PF_PATH: [], PF_ORDERBY: int( self._b_orderBy ), PF_GROUPBY: int( self._b_groupBy ), PF_LIMIT: int( self._b_limit ), \
                        PF_WHERE: int( self._b_where ), PF_HAVING: int( self._b_having ), GF_NUMCOL: 0, GF_COLLIST: [], \
                        OF_DESCRIPTOR: 0, OF_NUMVU: 0, OF_VU_OPERATOR: [], OF_VU_AGG1: [], OF_VU_AGG2: [], OF_VU_COL1: [], OF_VU_COL2: [], OF_VU_DIST1: [], OF_VU_DIST2: [], \
                        SF_DISTINCT: int( self._info_select.is_distinct ), SF_NUM_VU: len( self._info_select.vec_select_target ), \
                        SF_VU_AGGALL: [ v[0] for v in self._info_select.vec_select_target ], \
                        SF_VU_OPERATOR: [ v[1].operator_idx for v in self._info_select.vec_select_target ], \
                        SF_VU_COL1: [  v[1].col_left.column.col_idx for v in self._info_select.vec_select_target ], \
                        SF_VU_AGG1: [ v[1].col_left.aggregator_idx for v in self._info_select.vec_select_target ], \
                        SF_VU_DIST1: [ v[1].col_left.is_distinct for v in self._info_select.vec_select_target ], \
                        SF_VU_COL2: [ v[1].col_right.column.col_idx if v[1].col_right != None else 0 for v in self._info_select.vec_select_target ], \
                        SF_VU_AGG2: [ v[1].col_right.aggregator_idx if v[1].col_right != None else 0 for v in self._info_select.vec_select_target ], \
                        SF_VU_DIST2: [ v[1].col_right.is_distinct if v[1].col_right != None else 0 for v in self._info_select.vec_select_target ], \
                        LF_ISMAX: False, LF_POINTERLOC: 0, \
                        WF_NUM_CONDUNIT: 0, WF_CU_AGGREGATOR: [], WF_CU_IS_NOT: [], WF_CU_COND_OP: [], WF_CU_VAL1_TYPE: [], WF_CU_VAL1_BOOLVAL: [], WF_CU_VAL1_LIKELY: [], WF_CU_VAL1_SP: [], WF_CU_VAL1_EP: [], \
                        WF_CU_VAL2_TYPE: [],  WF_CU_VAL2_SP: [], WF_CU_VAL2_EP: [], WF_CU_VAL2_BOOLVAL: [], WF_CU_VAL2_LIKELY: [], \
                        WF_CU_VU_OPERATOR: [], WF_CU_VU_AGG1: [], WF_CU_VU_COL1: [], WF_CU_VU_DIST1: [], WF_CU_VU_AGG2: [], WF_CU_VU_COL2: [], WF_CU_VU_DIST2: [], \
                        WF_CU_VAL1_IGNORE: [], WF_CU_VAL2_IGNORE: [], \
                        HV_NUM_CONDUNIT: 0, HV_CU_AGGREGATOR: [], HV_CU_IS_NOT: [], HV_CU_COND_OP: [], HV_CU_VAL1_TYPE: [], HV_CU_VAL1_BOOLVAL: [], HV_CU_VAL1_LIKELY: [], HV_CU_VAL1_SP: [], HV_CU_VAL1_EP: [], \
                        HV_CU_VAL2_TYPE: [],  HV_CU_VAL2_SP: [], HV_CU_VAL2_EP: [], HV_CU_VAL2_BOOLVAL: [], HV_CU_VAL2_LIKELY: [], \
                        HV_CU_VU_OPERATOR: [], HV_CU_VU_AGG1: [], HV_CU_VU_COL1: [], HV_CU_VU_DIST1: [], HV_CU_VU_AGG2: [], HV_CU_VU_COL2: [], HV_CU_VU_DIST2: [], \
                        HV_CU_VAL1_IGNORE: [], HV_CU_VAL2_IGNORE: [] }

        if self._b_groupBy:
            features[ GF_NUMCOL ]   = len( self._info_groupBy )
            features[ GF_COLLIST ]  = [ c.col_idx for c in self._info_groupBy ] 

        if self._b_orderBy:
            features[ OF_DESCRIPTOR ]   = self._info_orderBy.desc_idx
            features[ OF_NUMVU ]        = len( self._info_orderBy.vec_value_unit )
            features[ OF_VU_OPERATOR ]  = [ vu.operator_idx for vu in self._info_orderBy.vec_value_unit ]
            features[ OF_VU_COL1 ]      = [ vu.col_left.column.col_idx for vu in self._info_orderBy.vec_value_unit ]
            features[ OF_VU_AGG1 ]      = [ vu.col_left.aggregator_idx for vu in self._info_orderBy.vec_value_unit ]
            features[ OF_VU_DIST1 ]     = [ vu.col_left.is_distinct for vu in self._info_orderBy.vec_value_unit ]
            features[ OF_VU_COL2 ]      = [ vu.col_right.column.col_idx if vu.col_right != None else 0 for vu in self._info_orderBy.vec_value_unit ]
            features[ OF_VU_AGG2 ]      = [ vu.col_right.aggregator_idx if vu.col_right != None else 0 for vu in self._info_orderBy.vec_value_unit ]
            features[ OF_VU_DIST2 ]     = [ vu.col_right.is_distinct if vu.col_right != None else 0 for vu in self._info_orderBy.vec_value_unit ]

        if self._b_limit:
            features[ LF_ISMAX ]        = self._limit_pos == -1
            features[ LF_POINTERLOC ]   = self._limit_pos if not features[ LF_ISMAX ] else 0 

        if self._b_where: 
            features[ WF_NUM_CONDUNIT ]     = len( self._info_where.vec_condunits )
            features[ WF_CU_AGGREGATOR ]    = [ v.aggregator for v in self._info_where.vec_condunits ]
            features[ WF_CU_IS_NOT ]        = [ v.is_not for v in self._info_where.vec_condunits ]
            features[ WF_CU_COND_OP ]       = [ v.cond_op_id for v in self._info_where.vec_condunits ] 
            features[ WF_CU_VAL1_TYPE ]     = [ v.val1_qtype for v in self._info_where.vec_condunits ]
            features[ WF_CU_VAL1_BOOLVAL ]  = [ v.val1_boolval for v in self._info_where.vec_condunits ]
            features[ WF_CU_VAL1_LIKELY ]   = [ v.val1_like for v in self._info_where.vec_condunits ]
            features[ WF_CU_VAL1_SP ]       = [ v.val1_sp for v in self._info_where.vec_condunits ]
            features[ WF_CU_VAL1_EP ]       = [ v.val1_ep for v in self._info_where.vec_condunits ]
            features[ WF_CU_VAL2_TYPE ]     = [ v.val2_qtype for v in self._info_where.vec_condunits ]
            features[ WF_CU_VAL2_BOOLVAL ]  = [ v.val2_boolval for v in self._info_where.vec_condunits ]
            features[ WF_CU_VAL2_LIKELY ]   = [ v.val2_like for v in self._info_where.vec_condunits ]
            features[ WF_CU_VAL2_SP ]       = [ v.val2_sp for v in self._info_where.vec_condunits ]
            features[ WF_CU_VAL2_EP ]       = [ v.val2_ep for v in self._info_where.vec_condunits ]
            features[ WF_CU_VU_OPERATOR ]   = [ v.value_unit.operator_idx for v in self._info_where.vec_condunits ]
            features[ WF_CU_VU_AGG1 ]       = [ v.value_unit.col_left.aggregator_idx for v in self._info_where.vec_condunits ]
            features[ WF_CU_VU_COL1 ]       = [ v.value_unit.col_left.column.col_idx for v in self._info_where.vec_condunits ]
            features[ WF_CU_VU_DIST1 ]      = [ v.value_unit.col_left.is_distinct for v in self._info_where.vec_condunits ] 
            features[ WF_CU_VU_AGG2 ]       = [ v.value_unit.col_right.aggregator_idx if v.value_unit.col_right != None else 0 for v in self._info_where.vec_condunits ]
            features[ WF_CU_VU_COL2 ]       = [ v.value_unit.col_left.column.col_idx if v.value_unit.col_right != None else 0 for v in self._info_where.vec_condunits ]
            features[ WF_CU_VU_DIST2 ]      = [ v.value_unit.col_left.is_distinct if v.value_unit.col_right != None else 0 for v in self._info_where.vec_condunits ] 
            
            features[ WF_CU_VAL1_IGNORE ]   = [ v.val1_ignore for v in self._info_where.vec_condunits ]
            features[ WF_CU_VAL2_IGNORE ]   = [ v.val2_ignore for v in self._info_where.vec_condunits ]
       
        if self._b_having: 
            features[ HV_NUM_CONDUNIT ]     = len( self._info_having.vec_condunits )
            features[ HV_CU_AGGREGATOR ]    = [ v.aggregator for v in self._info_having.vec_condunits ]
            features[ HV_CU_IS_NOT ]        = [ v.is_not for v in self._info_having.vec_condunits ]
            features[ HV_CU_COND_OP ]       = [ v.cond_op_id for v in self._info_having.vec_condunits ] 
            features[ HV_CU_VAL1_TYPE ]     = [ v.val1_qtype for v in self._info_having.vec_condunits ]
            features[ HV_CU_VAL1_BOOLVAL ]  = [ v.val1_boolval for v in self._info_having.vec_condunits ]
            features[ HV_CU_VAL1_LIKELY ]   = [ v.val1_like for v in self._info_having.vec_condunits ]
            features[ HV_CU_VAL1_SP ]       = [ v.val1_sp for v in self._info_having.vec_condunits ]
            features[ HV_CU_VAL1_EP ]       = [ v.val1_ep for v in self._info_having.vec_condunits ]
            features[ HV_CU_VAL2_TYPE ]     = [ v.val2_qtype for v in self._info_having.vec_condunits ]
            features[ HV_CU_VAL2_BOOLVAL ]  = [ v.val2_boolval for v in self._info_having.vec_condunits ]
            features[ HV_CU_VAL2_LIKELY ]   = [ v.val2_like for v in self._info_having.vec_condunits ]
            features[ HV_CU_VAL2_SP ]       = [ v.val2_sp for v in self._info_having.vec_condunits ]
            features[ HV_CU_VAL2_EP ]       = [ v.val2_ep for v in self._info_having.vec_condunits ]
            features[ HV_CU_VU_OPERATOR ]   = [ v.value_unit.operator_idx for v in self._info_having.vec_condunits ]
            features[ HV_CU_VU_AGG1 ]       = [ v.value_unit.col_left.aggregator_idx for v in self._info_having.vec_condunits ]
            features[ HV_CU_VU_COL1 ]       = [ v.value_unit.col_left.column.col_idx for v in self._info_having.vec_condunits ]
            features[ HV_CU_VU_DIST1 ]      = [ v.value_unit.col_left.is_distinct for v in self._info_having.vec_condunits ] 
            features[ HV_CU_VU_AGG2 ]       = [ v.value_unit.col_right.aggregator_idx if v.value_unit.col_right != None else 0 for v in self._info_having.vec_condunits ]
            features[ HV_CU_VU_COL2 ]       = [ v.value_unit.col_left.column.col_idx if v.value_unit.col_right != None else 0 for v in self._info_having.vec_condunits ]
            features[ HV_CU_VU_DIST2 ]      = [ v.value_unit.col_left.is_distinct if v.value_unit.col_right != None else 0 for v in self._info_having.vec_condunits ] 
            
            features[ HV_CU_VAL1_IGNORE ]   = [ v.val1_ignore for v in self._info_having.vec_condunits ]
            features[ HV_CU_VAL2_IGNORE ]   = [ v.val2_ignore for v in self._info_having.vec_condunits ]

        features[ META_DB ] = self._db

        # Used table info.
        table_used      = set( self._info_from_tbl ) | self._recover_tables( features )
        table_used_idx  = [ t.tbl_idx for t in table_used ]
        features[ TV_TABLES_USED_IDX ]  = table_used_idx
        features[ TV_TABLES_NUM ]       = len( table_used_idx )
        
        vec_ret.append( features )

        # 2. Nested Subqueries.
        #    Subqueries are allowed only for WHERE and HAVING.
        if self._b_where:
            where_sels  = self._info_where.get_all_selections() # Vector of selection queries.
            for sidx, s in enumerate( where_sels ):
                vec_subq_features   = s.get_prop_cla_features()
                for sub_f in vec_subq_features:
                    sub_f[ PF_PATH ]    = [ PATH_WHERE ] + sub_f[ PF_PATH ]
                    for _ in range( sidx ):
                        sub_f[ PF_PATH ]    = [ PATH_PAR ] + sub_f[ PF_PATH ]
                        sub_f[ PF_PATH ]    = [ PATH_WHERE ] + sub_f[ PF_PATH ]

                    vec_ret.append( sub_f )

        if self._b_having:
            having_sels  = self._info_having.get_all_selections() # Vector of selection queries.
            for sidx, s in enumerate( having_sels ):
                vec_subq_features   = s.get_prop_cla_features()
                for sub_f in vec_subq_features:
                    sub_f[ PF_PATH ]    = [ PATH_HAVING ] + sub_f[ PF_PATH ]
                    for _ in range( sidx ):
                        sub_f[ PF_PATH ]    = [ PATH_PAR ] + sub_f[ PF_PATH ]
                        sub_f[ PF_PATH ]    = [ PATH_HAVING ] + sub_f[ PF_PATH ]

                    vec_ret.append( sub_f )
        
        return vec_ret

    def get_query( self ):
        q_str = "SELECT %s " % self._info_select.get_query()
        tbl_str = ""
        for tbl_idx, tbl in enumerate( self._info_from_tbl ):
            tbl_str += tbl.tbl_name_orig
            if tbl_idx < len( self._info_from_tbl ) - 1:
                tbl_str += ", "

        q_str += "FROM %s " % tbl_str
        if self._b_where:
            q_str   += "WHERE %s " % self._info_where.get_query()
        if self._b_having:
            q_str   += "HAVING %s " % self._info_having.get_query()
        if self._b_orderBy:
            q_str   += "ORDER BY %s " % self._info_orderBy.get_query()
        if self._b_groupBy:
            str_cols    = ""
            for cidx, c in enumerate( self._info_groupBy ):
                str_cols    += c.vec_cols[0].col_name_orig
                if cidx < len( self._info_groupBy ) - 1:
                    str_cols    += ", "
            q_str   += "GROUP BY %s " % str_cols
        if self._b_limit:
            q_str   += "LIMIT %s " % self._num_limit

        return q_str

    def _recover_tables( self, info ):
        db  = info[ META_DB ]

        # Gather used columns.
        set_tbls        = set()
        col_idx_meta    = [ GF_COLLIST, OF_VU_COL1, OF_VU_COL2, SF_VU_COL1, SF_VU_COL2, WF_CU_VU_COL1, WF_CU_VU_COL2, HV_CU_VU_COL1, HV_CU_VU_COL2 ]
        for col_meta_name in col_idx_meta:
            for c in info[ col_meta_name ]:
                orig_col    = db.vec_cols[ c ]
                if orig_col.table_belong != None:
                    set_tbls.add( orig_col.table_belong )

        return set_tbls

    
    # Parse ORDERBY PHRASE.
    def _parse_orderBy( self, qjson, db ):
        if qjson[ "orderBy" ] == []:
            self._b_orderBy     = False
            self._info_orderBy  = None
            return
        
        self._b_orderBy     = True
        self._info_orderBy  = Info_OrderBy()
        self._info_orderBy.read_from_json( qjson[ "orderBy" ], db )

    # Parse GROUPBY phrase.
    # Vector of Columns.
    def _parse_groupBy( self, qjson, db ):
        if qjson[ "groupBy" ] == []:
            self._b_groupBy     = False
            self._info_groupBy  = []
            return

        self._b_groupBy     = True
        self._info_groupBy  = []
        for c_info in qjson[ "groupBy" ]:
            self._info_groupBy.append( db.vec_cols[ c_info[1] ] )

    # Parse Limit.
    # Integer Number.
    def _parse_limit( self, qjson, qtoks ):
        if qjson[ "limit" ] == None:
            self._b_limit   = False
            self._num_limit = 0
            return
        
        self._b_limit   = True
        self._num_limit = int( qjson[ "limit" ] )
        vec_nums    = [ "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten" ]

        for tidx, t in enumerate( qtoks ):
            if self._num_limit != 1 and ( t == str( self._num_limit ) or vec_nums[ self._num_limit - 1 ] == t ):
                self._limit_pos = tidx

    # Parse WHERE Conditions.
    def _parse_wherecond( self, qjson, qtoks, db ):
        if qjson[ "where" ] == []:
            self._b_where   = False
            self._info_where    = None
            return

        self._b_where       = True
        self._info_where    = Info_Condition()
        self._info_where.read_from_json( qjson[ "where" ], qtoks, db )

    # Same as "where".
    def _parse_having( self, qjson, qtoks, db ):
        if qjson[ "having" ] == []:
            self._b_having      = False
            self._info_having   = None
            return

        self._b_having      = True
        self._info_having   = Info_Condition()
        self._info_having.read_from_json( qjson[ "having" ], qtoks, db )

    def _parse_select( self, qjson, db ):
        self._info_select   = Info_Select()
        self._info_select.read_from_json( qjson[ "select" ], db )

    def _parse_from( self, qjson, db ):
        self._info_from_tbl = []
        for tbl_info in qjson[ "from" ][ "table_units" ]:
            if tbl_info[ 0 ] == "table_unit":
                self._info_from_tbl.append( db.vec_tbls[ tbl_info[1] ] )

