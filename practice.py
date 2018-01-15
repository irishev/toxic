import csv
import re

f = open('C:/Users/dilab3/Downloads/stanfordSentimentTreebank/stanfordSentimentTreebank/datasetSentences.txt', 'r', encoding='utf-8')
rdr = csv.reader(f, delimiter='\t')

for i, line in enumerate(rdr):
    if i!=0:
        words = re.sub("[^\w]", " ",  str.lower(line[1])).split()
        wordlist = []
        for word in words:
            if word in word2ind.keys():
                wordlist.append(word2ind[word])
            else:
                wordlist.append(0)
        if len(wordlist) < 64:
            for _ in range(64-len(wordlist)):
                wordlist.append(0)
        elif len(wordlist) > 64:
            wordlist = wordlist[0:64]

        if sum(wordlist)==0:
            print(line[1])
            continue

        target = line[2:]
        target = list(map(float, target))

        dataset.append((wordlist,torch.FloatTensor(target)))

f = open('C:/Users/dilab3/Downloads/stanfordSentimentTreebank/stanfordSentimentTreebank/sentiment_labels.txt', 'r', encoding='utf-8')
rdr = csv.reader(f, delimiter='|')

for i, line in enumerate(rdr):
    print(line)

f.close()
