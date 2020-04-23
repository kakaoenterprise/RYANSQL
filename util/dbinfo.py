import sys

from nltk.stem      import PorterStemmer
from nltk.tokenize  import TreebankWordTokenizer

stemmer  = PorterStemmer()

class ColumnInfo:
    def __init__( self, col_name, col_name_orig, col_type ):
        self.col_name       = col_name
        self.col_name_orig  = col_name_orig     
        self.col_type       = col_type
        self.col_idx        = -1
        self.is_primary     = False
        self.is_foreign     = False
        self.table_belong   = None

        self.vec_norm_name  = [ stemmer.stem( v.lower() ) for v in TreebankWordTokenizer().tokenize( self.col_name ) ]
        self.vec_col_rep    = TreebankWordTokenizer().tokenize( self.col_name )

    def get_rep_name( self, cn_type = 0 ):
        if cn_type == 0:
            return self.vec_col_rep

        if cn_type == 1:
            if self.table_belong == None:
                return self.vec_col_rep
            vec_tbl_rep = self.table_belong.get_rep_name()
            vec_ret = []
            vec_tbl_add = []
            for t, n in zip( self.table_belong.get_rep_name(), self.table_belong.vec_norm_name ):
                if n not in self.vec_norm_name:
                    vec_tbl_add.append( t )

            vec_ret = vec_tbl_add + self.vec_col_rep

            return vec_ret

        if cn_type == 2:
            if self.table_belong == None:
                return self.vec_col_rep

            tbl = self.table_belong

            # 1. Check if the expansion is required.
            req_exp     = False
            vec_comp_c  = []
            for c in tbl.db.vec_cols:
                if c == self:
                    continue

                all_in  = True
                for r in self.vec_col_rep:
                    if r not in c.vec_col_rep:
                        all_in  = False
                        break
                if all_in:
                    # Check if the two columns represent the same info.
                    is_foreign  = False
                    for f1_idx, f2_idx in tbl.db.foreign_keys:
                        if ( f1_idx == self and f2_idx == c ) or ( f1_idx == c and f2_idx == self ):
                            is_foreign  = True
                            break
                    if is_foreign == False or ( is_foreign and self.col_name != c.col_name and self.col_name in c.col_name ):
                        req_exp = True
            
            # 2. Expand col name if required.
            if req_exp:
                vec_tbl_rep = self.table_belong.get_rep_name()
                vec_ret = []
                vec_tbl_add = []
                for t, n in zip( self.table_belong.get_rep_name(), self.table_belong.vec_norm_name ):
                    if n not in self.vec_norm_name:
                        vec_tbl_add.append( t )

                vec_ret = vec_tbl_add + self.vec_col_rep
                return vec_ret
            return self.vec_col_rep

    def mark_primary( self ):
        self.is_primary = True
        self.table_belong.vec_primary_keys.append( self )
     
    def mark_foreign( self ):
        self.is_foreign = True

    def print_column( self ):
        print ( "%s (%s): %s" % ( self.col_name, self.col_name_orig, self.col_type ) )

class TableInfo:
    def __init__( self, tbl_name, tbl_name_orig ):
        self.tbl_name           = tbl_name
        self.tbl_name_orig      = tbl_name_orig
        self.tbl_idx            = -1
        self.vec_cols           = []
        self.vec_primary_keys   = []
        self.db                 = None

        self.vec_norm_name      = [ stemmer.stem( v.lower() ) for v in self.get_rep_name() ]

    def get_rep_name( self ):
        return TreebankWordTokenizer().tokenize( self.tbl_name )

    def set_columns( self, vec_cols ):
        self.vec_cols       = vec_cols
        for v in self.vec_cols:
            if v.table_belong != None:
                print ( "SINGLE COLUMN CANNOT BELONG TO MORE THAN TWO TABLES" )
                sys.exit(0) 
            v.table_belong  = self

    def print_table( self ):
        print ( "TABLE %s (%s)" % ( self.tbl_name, self.tbl_name_orig ) )
        if len( self.vec_primary_keys ) > 0:
            print ( "PRIMARY KEY: %s" % ( ",".join( [ k.col_name_orig for k in self.vec_primary_keys ] ) ) )
        print ( "-------- COLUMNS --------" )
        for c in self.vec_cols:
            c.print_column()


    
class DBInfo:
    # Expected format of db_info_dic
    # db_id: database id.
    # table_names: vector of table names.
    # table_names_original: vector of original table names.
    # column_names: vector of [ table index - column names ]. 0-th column is padding for "count all" things.
    # column_names_original
    # column_types: types of the column. text / time / number / others
    # primary_keys: index of primary keys.
    # foreign_keys: vector of foreign key pairs.
    # Example of a data:
    #
    # db_id: product_catalog
    # column_names: [[-1, '*'], [0, 'attribute id'], [0, 'attribute name'], [0, 'attribute data type'], [1, 'catalog id'], [1, 'catalog name'], [1, 'catalog publisher'], [1, 'date of publication'], [1, 'date of latest revision'], [2, 'catalog level number'], [2, 'catalog id'], [2, 'catalog level name'], [3, 'catalog entry id'], [3, 'catalog level number'], [3, 'parent entry id'], [3, 'previous entry id'], [3, 'next entry id'], [3, 'catalog entry name'], [3, 'product stock number'], [3, 'price in dollars'], [3, 'price in euros'], [3, 'price in pounds'], [3, 'capacity'], [3, 'length'], [3, 'height'], [3, 'width'], [4, 'catalog entry id'], [4, 'catalog level number'], [4, 'attribute id'], [4, 'attribute value']]
    # primary_keys: [1, 4, 9, 12]
    # column_types: ['text', 'number', 'text', 'text', 'number', 'text', 'text', 'time', 'time', 'number', 'number', 'text', 'number', 'number', 'number', 'number', 'number', 'text', 'text', 'number', 'number', 'number', 'text', 'text', 'text', 'text', 'number', 'number', 'number', 'text']
    # foreign_keys: [[10, 4], [13, 9], [27, 9], [26, 12]]
    # column_names_original: [[-1, '*'], [0, 'attribute_id'], [0, 'attribute_name'], [0, 'attribute_data_type'], [1, 'catalog_id'], [1, 'catalog_name'], [1, 'catalog_publisher'], [1, 'date_of_publication'], [1, 'date_of_latest_revision'], [2, 'catalog_level_number'], [2, 'catalog_id'], [2, 'catalog_level_name'], [3, 'catalog_entry_id'], [3, 'catalog_level_number'], [3, 'parent_entry_id'], [3, 'previous_entry_id'], [3, 'next_entry_id'], [3, 'catalog_entry_name'], [3, 'product_stock_number'], [3, 'price_in_dollars'], [3, 'price_in_euros'], [3, 'price_in_pounds'], [3, 'capacity'], [3, 'length'], [3, 'height'], [3, 'width'], [4, 'catalog_entry_id'], [4, 'catalog_level_number'], [4, 'attribute_id'], [4, 'attribute_value']]
    # table_names_original: ['Attribute_Definitions', 'Catalogs', 'Catalog_Structure', 'Catalog_Contents', 'Catalog_Contents_Additional_Attributes']
    # table_names: ['attribute definitions', 'catalogs', 'catalog structure', 'catalog contents', 'catalog contents additional attributes']
    def __init__( self, db_info_dic ):
        # 1. Read basic info for columns and tables.
        self.db_id      = db_info_dic[ "db_id" ]
        self.vec_cols   = [ ColumnInfo( col_name[1], col_name_orig[1], col_type ) \
                            for col_name, col_name_orig, col_type \
                            in zip( db_info_dic[ "column_names" ], db_info_dic[ "column_names_original" ], db_info_dic[ "column_types" ] ) ]    # Column 0 is always *.

        for cidx, c in enumerate( self.vec_cols ):
            c.col_idx   = cidx
        
        self.vec_tbls   = [ TableInfo( refined_name, orig_name ) for orig_name, refined_name in zip( db_info_dic[ "table_names_original" ], db_info_dic[ "table_names" ] ) ]
        for tidx, t in enumerate( self.vec_tbls ):
            t.tbl_idx   = tidx

        self.foreign_keys   = []
        
        # 2. Assign columns to each table.
        cur_table_idx   = -1
        col_start       = -1
        vec_col_info    = []
        for idx, col_info in enumerate( db_info_dic[ "column_names" ][ 1: ] ):
            if col_info[0] != cur_table_idx:
                if cur_table_idx != -1:
                    vec_col_info.append( [ col_start + 1, idx + 1 ] )
                col_start       = idx
                cur_table_idx   = col_info[0]
        vec_col_info.append( [ col_start + 1, len( self.vec_cols ) ] )

        for tbl, col_span in zip( self.vec_tbls, vec_col_info ):
            tbl.set_columns( self.vec_cols[ col_span[0]:col_span[1] ] )     
        
        # 3. Mark primary keys.
        for primary_key_idx in db_info_dic[ "primary_keys" ]:
            self.vec_cols[ primary_key_idx ].mark_primary()

        # 4. Set foreign key info.
        for f1, f2 in db_info_dic[ "foreign_keys" ]:
            self.foreign_keys.append( [ self.vec_cols[ f1 ], self.vec_cols[ f2 ] ] )
            self.vec_cols[ f1 ].mark_foreign() 

        for t in self.vec_tbls:
            t.db    = self

    def print_dbinfo( self ):
        print ( "--------------------------------------------------------------------------------" )
        print ( "DATABASE ID: %s" % self.db_id )
        print ( "COLUMNS: " )
        for cidx, c in enumerate( self.vec_cols ):
            print ( "COL [%d]: " % cidx, end = "" )
            c.print_column()

        print ( "TABLES:" )
        for t in self.vec_tbls:
            print ( "=======================================================" )
            t.print_table()
        print ( "============ FOREIGN KEYS =============================" )
        for c1, c2 in self.foreign_keys:
            print ( "%s : %s <-----> %s : %s " % ( c1.table_belong.tbl_name, c1.col_name, c2.table_belong.tbl_name, c2.col_name ) )
        

    

