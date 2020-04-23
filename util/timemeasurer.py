
from __future__ import print_function
import datetime as dt

class TimeMeasurer:
    def __init__( self ):
        self.startdic = dict()
        self.elapseddic = dict()

    def start( self, key ):
        self.startdic[ key ] = dt.datetime.now()

    def end( self, key ):
        if key not in self.startdic:
            print ( "NO START TIME SIGNAL"  )
            return
        endtime = dt.datetime.now()
        elapsed = (endtime - self.startdic[key]).total_seconds() * 1000
        if key in self.elapseddic:
            self.elapseddic[ key ] += elapsed
        else:
            self.elapseddic[ key ] = elapsed

    def printitem( self, key ):
        if key not in self.elapseddic:
            print ( "Not such key in time elapsed dic" )
            return

        print ( "Time for %s: %d msec" % ( key, self.elapseddic[ key ] ) )
    
    def getElapsed( self, key ):
        if key not in self.elapseddic:
            print ( "Not such key in time elapsed dic" )
            return 0

        return self.elapseddic[ key ]
        
    def printall( self ):
        for k, v in self.elapseddic.items():
            print ( "Time for %s: %d msec" % ( k, v ) )

