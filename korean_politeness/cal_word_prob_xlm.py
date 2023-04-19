import sys
import numpy as np

data = []
for line in open(sys.argv[1]):
    line = line.strip().split()
    if len(line):
        data.append(line)

out = open(sys.argv[2], 'w')
for sent_ind in range(int(len(data)/2)):
    sent1 = data[sent_ind*2]
    print(sent1)
    sent11 = [sent1[0]]
    for w in sent1[1:]:
        if not w[0] == '▁':
            sent11.append('##'+w)
        else:
            sent11.append(w[1:])
    sent1 = sent11

    sent2 = [float(i) for i in data[sent_ind*2+1]]
    mid = []
    for ind,word in enumerate(sent1):
        if word[0:2] == '##':
            mid2 = [ind-1]
            for ind2 in range(ind, len(sent1)):
                if sent1[ind2][0:2] == '##':
                    mid2.append(ind2)
                else:
                    break
            if len(set(mid2)&set([i for sublist in mid for i in sublist])):
                continue
            else:
                mid.append(mid2)
    print(mid)
    out.write(' '.join(sent1)+'\n')
    prob_mid = 1
    midindex = [i[0] for i in mid]
    print(midindex)
    ind = 0
    while ind < len(sent2):
        if not ind in midindex:
            print(sent2[ind])
            out.write(str(sent2[ind])+' ')
            ind += 1
        else:
            for i in mid[midindex.index(ind)]:
                prob_mid = prob_mid*sent2[ind]*sent2[i]
                ind += 1
            print(prob_mid)
            out.write(str(prob_mid)+' ')
    out.write('\n')
