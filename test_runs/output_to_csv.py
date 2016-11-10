import sys
import re


for line in open(sys.argv[1]):
    if '|' in line:
        print ','.join(num for num in re.findall('\d+', line))
