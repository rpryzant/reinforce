"""
this file does some quick number crunching

"""
import sys


s = 0

for file in sys.argv[2:]:
    s += float(open(file).readlines()[-1].split(',')[2])

print s

