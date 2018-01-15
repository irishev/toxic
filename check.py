import csv

f = open('C:/Users/dilab3/Downloads/6339009/submission.csv', 'r', encoding='utf-8')
rdr = csv.reader(f)

for i, line in enumerate(rdr):
    if (len(line) != 7) & (i%2==0):
        print(line)
        print("error at %d line" % (i))
