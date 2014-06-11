__author__ = 'LIMU_North'

import sqlite3 as lite
import sys

con = None

print lite.version
print lite.sqlite_version

try:
    con = lite.connect('C:/Cassandra/test.db')
    cur = con.cursor()
    data = cur.fetchone()
    print "SQLite version: %s" %data

except lite.Error, e:
    print "Error %s:" % e.args[0]
    sys.exit(1)

finally:
    if con:
        con.close()