import sys
from itertools import combinations
import time

def apriori(large, support):
    #Candidates generated from large
    if n == 1:
        candidate = list(combinations(large, 2))
    else:
        candidate = {}
        for i in large:
            for j in i:
                if j not in candidate.keys():
                    candidate[j] = 0
        candidate = set(combinations(candidate.keys(), n+1))

    #Make large from candidates with min_support
    candidate_k = {}
    if n == 1:
        tmp = []
        for i in large:
            tmp.append(list([i]))
        large = tmp
    else:
        tmp = []
        for i in large:
            tmp.append(set(i))
        large = tmp

    #Delete list that not in previous large
    del_candidate_k = [] 
    for i in candidate:
        candidate_k[i] = 0
        for j in list(combinations(i, n)):
            if n == 1:
                j = list(j)
            else:
                j = set(j)

            if j not in large:
                del_candidate_k.append(i)
                break
    for i in del_candidate_k:
        candidate_k.pop(i)

    #DB scan
    for i in data:
        for j in candidate_k:
            if set(j).issubset(set(i)):
                candidate_k[j] += 1

    #Min_support
    new_large = {}
    for i in candidate_k.keys():
        if candidate_k[i] >= support:
            new_large[i] = candidate_k[i]

    return new_large
    
def result_write(large):
    global total
    ff = open(sys.argv[3], mode='a', encoding='utf-8')
    num = n
    for i, j in large.items():
        for k in range(num):
            tmp = list(combinations(i, k+1))
            for l in tmp:
                temp = l
                l = set(l)

                ff.write('{')
                for m in range(len(temp)):
                    if m == len(temp)-1:
                        ff.write(str(temp[m]))
                    else:
                        ff.write(str(temp[m]))
                        ff.write(',')
                ff.write('}')
                ff.write('\t')
                temp = list(set(i) - l)
                ff.write('{')
                for m in range(len(temp)):
                    if m == len(temp)-1:
                        ff.write(str(temp[m]))
                    else:
                        ff.write(str(temp[m]))
                        ff.write(',')
                ff.write('}')
                ff.write('\t')
                ff.write("{0:0.2f}".format(round(j / data_num * 100, 2)))
                ff.write('\t')            
                check = 0
                for m in data:
                    if l.issubset(set(m)):
                        check += 1
                ff.write("{0:0.2f}\n".format(round(j / check * 100, 2)))
                total += 1
    ff.close()


global total
data = []
n = 1
total = 0
start = time.time()

#Input data
f = open(sys.argv[2], 'r')
lines = f.read().split('\n')
for line in lines:
    line = line.split('\t')
    tmp = []
    for i in line:
        tmp.append(int(i))
    data.append(tmp)

#Make first frequent items
candidate_1 = {}
for i in data:
    for j in i:
        if j in candidate_1.keys():
            candidate_1[j] += 1
        else:
            candidate_1[j] = 1
            
data_num = len(data)
support = float(sys.argv[1]) / 100 * data_num
large_k = {}
for i in candidate_1.keys():
    if candidate_1[i] >= support:
        large_k[i] = candidate_1[i]
large_k = [large_k]

#Start Apriori algorithm
while(1):
    large = list(large_k[n-1].keys())
    new_large = apriori(large, support)
    result_write(new_large)

    if new_large:
        large_k.append(new_large)
        n += 1
    else:
        print("<Complete>")
        print("Total result : {}".format(total))
        print("Time : {}".format(time.time() - start))
        break
