import sys
sys.path.append( "../../util" )

from db_meta    import *

class KeyInfo:
    def __init__( self, key, tensor, text_rep, check_cond = None, match_func = None, is_tbl = False ):
        self.key        = key
        self.tensor     = tensor
        self.text_rep   = text_rep
        self.is_tbl     = is_tbl

        self.num_cor        = 0
        self.num_tot        = 0
        if match_func == None:
            self.match_func = lambda data, result: data[ self.key ] == result[ self.key ]
        else:
            self.match_func = match_func
        self.prereq_cond    = check_cond       # Lambda prerequisite function, to test for the key. lambda data, eval.

class TestInfo:
    def __init__( self ):
        self.vec_key_info   = []

        self.all_tot    = 0
        self.all_cor    = 0

    def add_test_val( self, key, tensor, text_rep, cond = None, match_func = None, is_tbl = False ):
        key_info    = KeyInfo( key, tensor, text_rep, check_cond = cond, match_func = match_func, is_tbl = is_tbl )     
        self.vec_key_info.append( key_info )

    def fetch_tbl_tensor_info( self, sess, feed_dict ):
        return self.fetch_tensor_info( sess, feed_dict, [ v for v in self.vec_key_info if v.is_tbl ] )

    def fetch_prop_tensor_info( self, sess, feed_dict ):
        return self.fetch_tensor_info( sess, feed_dict, [ v for v in self.vec_key_info if v.is_tbl == False ] )

    # Returns: dictionary of [ Feature Key ] - [ Evaluated Value ].
    def fetch_tensor_info( self, sess, feed_dict, vec_tensor_fetch_list ):
        vec_retrieved           = sess.run( [ v.tensor for v in vec_tensor_fetch_list ], feed_dict = feed_dict )
        fetch_info  = dict()
        for key_info, retrieved in zip( vec_tensor_fetch_list, vec_retrieved ):
            fetch_info[ key_info.key ]  = retrieved.tolist()
        
        return fetch_info

    # Integrate the evaluation results.
    def integrate_eval_result( self, evaluated_values, vec_data ):
        for didx, data in enumerate( vec_data ):
            # Generate data for single data item.
            single_data_info    = dict()
            for key_info in self.vec_key_info:
                single_data_info[ key_info.key ]    = evaluated_values[ key_info.key ][ didx ]
        
            # Evaluate.
            wrong   = False
            for key_info in self.vec_key_info:
                if key_info.prereq_cond == None or key_info.prereq_cond( data, single_data_info ):
                    key_info.num_tot    += 1
                    if key_info.match_func( data, single_data_info ):
                        key_info.num_cor    += 1
                    else:
                        wrong   = True

            self.all_tot    += 1
            if wrong == False:
                self.all_cor    += 1

    def print_eval_results( self ):
        max_str_len = max( [ len( v.text_rep ) for v in self.vec_key_info ] )
        for key_info in self.vec_key_info:
            if key_info.num_tot == 0:
                continue
#                print ( "%s: 0 / 0" % ( key_info.text_rep.ljust( max_str_len ) ) )
            else:
                print ( "%s: %d / %d = %.2f" % ( key_info.text_rep.ljust( max_str_len ), key_info.num_cor, key_info.num_tot, key_info.num_cor / key_info.num_tot * 100.0 ) )
        
        print ( "%s: %d / %d = %.2f" % ( "ALL".ljust( max_str_len ), self.all_cor, self.all_tot, self.all_cor / self.all_tot * 100.0 ) )
    
    def get_overall_result( self ):
        return self.all_cor / self.all_tot * 100.0
            
