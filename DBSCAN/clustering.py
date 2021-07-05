import sys
import numpy as np
from collections import Counter
import time
import pandas as pd
import matplotlib.pyplot as plt

#Check epsilon
def check_epsilon(data, Pt, Eps):
    core_candidate = []
    tmp = np.array(data)
    data_xy = np.array(tmp[:, 1:])
    xy = np.array(Pt[1:])
    #Calculate distance    
    for i in range(len(data_xy)):
        if np.sqrt((data_xy[i][0] - xy[0])**2 + (data_xy[i][1] - xy[1])**2) < Eps:
            core_candidate.append(i)
            
    #return core_candidate
    return list(dict.fromkeys(core_candidate))

#Time check    
start = time.time()

#Input data
data = []

f = open(sys.argv[1], 'r')
lines = f.read().split('\n')
for line in lines:
    if line != '':
        line = line.split('\t')
        tmp = list(map(float, line))
        data.append(tmp)

n = int(sys.argv[2])
Eps = int(sys.argv[3])
MinPts = int(sys.argv[4])

labels = [-1] * len(data)
cluster_num = 0

for i in range(len(data)):
    #Find core candidate
    if labels[i] == -1:
        core_candidate = check_epsilon(data, data[i], Eps)
        #Directly density reachable
        if len(core_candidate) >= MinPts:
            #Labeling
            labels[i] = cluster_num
            #Density reachable
            j = 0
            while j < len(core_candidate):
                if labels[core_candidate[j]] == -1:
                    #labeling
                    labels[core_candidate[j]] = cluster_num
                    #Find density reachable
                    distance_reachable = check_epsilon(data, data[core_candidate[j]], Eps)

                    if len(distance_reachable) >= MinPts:
                        core_candidate += distance_reachable
                        list(dict.fromkeys(core_candidate))


                j += 1
                
            cluster_num += 1

        #Noise              
        else:
            labels[i] = -2


#(Optional) Sort for printing n output files
cnt = Counter(labels)
res = []
for i in cnt:
    if i >= 0:
        res.append([cnt[i], i])
res.sort(reverse = True)

#Output data
file_name = ""
for c in sys.argv[1]:
    if c != '.':
        file_name += c
    else:
        break

for i in range(n):
    ff = open(file_name + "_cluster_" + str(i) + ".txt", 'w')
    for j in range(len(labels)):
        if labels[j] == res[i][1]:
            ff.write(str(j))
            ff.write('\n')
    ff.close
        
print("time : {} sec\n".format(time.time() - start))

#For rubric image
rubric_data = pd.DataFrame(data)
unique_labels = set(labels)
colors = [plt.cm.gist_rainbow(each)
        for each in np.linspace(0, 1, len(unique_labels))]

plt.figure(figsize=[8, 8])
for cluster_index, col in zip(unique_labels, colors):
    if cluster_index == -1:
        col = [0, 0, 0, 1]
    class_mask = (np.array(labels) == cluster_index)
    plt.plot(rubric_data.values[class_mask][:, 1], 
             rubric_data.values[class_mask][:, 2], 
            'o', markerfacecolor=tuple(col), markeredgecolor=tuple(col), 
            markersize=1)
plt.show()
