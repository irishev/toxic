import json

# lines = open('content2.json').readlines()
'''
temp = {}
for line in lines:
    v = json.loads(line)
    s = v['topic']
    hier = s.split('/')
    parents = ""
    for n, h in enumerate(hier):
        if n==0:
            parents = h
        else:
            parents += '/' + h
        if parents in temp.keys():
            temp[parents] += 1
        else:
            temp[parents] = 1

f1 = open('ndesc.txt', mode='w', encoding='utf-8')
for k in temp.keys():
    f1.write(k + ':' + str(temp[k]) + '\n')

f1.close()
'''
'''
lines=open('ndesc.txt', mode='r', encoding='utf-8').readlines()
f = open('category.txt', mode='w', encoding='utf-8')
for line in lines:
    cat, n = line.split(':')
    catnames = cat.split('/')
    flag = True
    for name in catnames:
        if (len(name)==1) | (name=='Titles'):
            flag = False
    if flag:
        f.write(line)

f.close()
'''
'''
lines = open('category.txt', mode='r', encoding='utf-8').readlines()

s1 = set([])
for line in lines:
    parent = line[:line.rfind('/')]
    s1.add(line[:line.rfind(':')])
    if parent in s1:
        s1.remove(parent)

f = open('category2.txt', mode='w', encoding='utf-8')

for line in lines:
    cat, n = line.split(':')
    if (cat not in s1) and (int(n)>90):
        f.write(line)
f.close()

'''

lines = open('category2.txt').readlines()
pages = open('content2.json').readlines()

stat = {}
cpl = {}
ppl = {}
nvc = {}

for line in lines:
    cat, n = line.split(':')
    stat[cat] = 0
    cpl[cat.count('/')+1]=0

for page in pages:
    v = json.loads(page)
    s = v['topic']
    if s in stat.keys():
        cpl[s.count('/')+1]+=1
        stat[s]+=1

for k, v in stat.items():
    if k.count('/')+1 in ppl.keys():
        ppl[k.count('/')+1] += 1
    else:
        ppl[k.count('/')+1] = 1
    if v//5 in nvc.keys():
        nvc[v//5] += 1
    else:
        nvc[v//5] = 1

open('ppl.json', mode='w').write(json.dumps(ppl))
open('cpl.json', mode='w').write(json.dumps(cpl))
open('nvc.json', mode='w').write(json.dumps(nvc))
