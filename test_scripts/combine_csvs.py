
import collections
import sys


print sys.argv


files = collections.defaultdict(str)


for file in sys.argv[1:]:
    file_type = '-'.join(file.split('-')[2:])
    last_line = open(file).readlines()[-1]
    d[file_type] += last_line + '\n'

for filename, csv in d:
    f = open(filename, 'w')
    f.write(csv)
    f.close()

