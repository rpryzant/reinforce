
import collections
import sys


print sys.argv

# make {run type => [cumulative scores]} mapping
files = collections.defaultdict(list)
for file in sys.argv[1:]:
    file_type = '-'.join(file.split('-')[2:])
    cumulative_score = open(file).readlines()[-1].split(',')[0]
    d[file_type].append(cumulative_score)

# make csv with cumulative scores for each run type as columns
score_matrix = d.items()
csv = ','.join(name for name, values in score_matrix) + '\n'
csv += '\n'.join(','.join(values[i] for name, values in score_matrix) for i in range(len(score_matrix[0][1])))

# write the csv
for filename, csv in d:
    f = open('replayMemory_output', 'w')
    f.write(csv)
    f.close()

