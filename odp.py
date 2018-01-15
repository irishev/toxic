import json
from tqdm import tqdm

dict = {}

lines = open('content.json').readlines()
for line in lines:
    dict[len(dict)] = json.loads(line)

lines = open('ad-content.json').readlines()
for line in lines:
    dict[len(dict)] = json.loads(line)

lines = open('kt-content.json').readlines()
for line in lines:
    dict[len(dict)] = json.loads(line)


list = []
for k, v in dict.items():
    if v['topic'][4:9]=='World':
        list.append(k)
    if v['topic'][6:11]=='World':
        list.append(k)
    if v['topic'][4:12]=='Regional':
        list.append(k)
    if v['topic'][6:14]=='Regional':
        list.append(k)
    if v['topic'][15:28]=='International':
        list.append(k)
    if v['topic'][0:3]=="Top":
        v['topic'] = v['topic'][4:]
    #if v['topic'].find('Titles') != -1:
    #    list.append(k)
    del v['url']

for k in list:
    dict.pop(k)

f = open('content2.json', mode='w')
for obj in dict.values():
    f.write(json.dumps(obj)+'\n')
