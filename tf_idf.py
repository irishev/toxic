import json

dictionary = {}

lines = open('content2.rdf', mode='r', encoding='utf-8')

for line in lines:
    v = json.loads(line)
    s = v['d:Description']
    words
    if s in stat.keys():
        cpl[s.count('/')+1]+=1
        stat[s]+=1
