import sys
import time
import pandas as pd
import numpy as np
from itertools import combinations

def entropy(column):
    #Count Each items
    items, cnt = np.unique(column, return_counts = True)
    info = 0
    #Calculate Pi and Entropy
    for i in range(len(items)):
        pi = cnt[i]/np.sum(cnt)
        info -= pi*np.log2(pi)
        
    return info

def gini(column):
    #Count Each items
    items, cnt = np.unique(column, return_counts = True)
    gini = 1
    for i in range(len(items)):
        pi = cnt[i]/np.sum(cnt)
        gini -= pi**2
        
    return gini

def gain(data, attr, target):
    #Expected information
    info = entropy(data[target])
    #Count Each items
    items, cnt = np.unique(data[attr], return_counts=True)
    #Calculate Information gained
    info_k = 0
    for i in range(len(items)):
        pi = cnt[i]/np.sum(cnt)
        subset = data.loc[data[attr] == items[i]]
        info_k += pi*entropy(subset[target])

    return info - info_k

def gain_ratio(data, attr, target):
    #Expected information
    info = entropy(data[target])
    #Count Each items
    items, cnt = np.unique(data[attr], return_counts=True)
    #Calculate Information gained and SplitInfo for gain ratio
    info_k = 0
    split_info = 0
    for i in range(len(items)):
        pi = cnt[i]/np.sum(cnt)
        subset = data.loc[data[attr] == items[i]]
        info_k += pi*entropy(subset[target])
        split_info -= pi*np.log2(pi)

    return (info - info_k) / split_info

def gini_index(data, attr, target):
    #Gini(D)
    gini_d = gini(data[target])
    #Count Each items
    items, cnt = np.unique(data[attr], return_counts=True)
    #Calculate Gini index
    tmp1 = 0
    tmp2 = 0
    gini_k = 0
    #More than two attributes (Need to group)
    if len(items) > 2:
        
        #Split attribute list
        #1 : Right larger
        #2 : Left larger
        n1 = int(len(items) / 2)
        n2 = int(len(items) / 2) + 1

        left1 = items[:n1]
        right1 = items[n1:]
        
        left2 = items[:n2]
        right2 = items[n2:]

        items1 = []
        items2 = []

        items1.append(left1)
        items1.append(right1)

        items2.append(left2)
        items2.append(right2)
        
        cnt1 = []
        cnt2 = []
        
        cnt1.append(np.sum(cnt[:n1]))
        cnt1.append(np.sum(cnt[n1:]))
        
        cnt2.append(np.sum(cnt[:n2]))
        cnt2.append(np.sum(cnt[n2:]))        

        for i in range(len(items1)):
            pi = cnt1[i]/np.sum(cnt1)
            subset1 = []
            #Concat subset with attributes
            for j in items1[i]:
                if len(subset1) == 0:
                    subset1 = data.loc[data[attr] == j]
                else:
                    subset1 = pd.concat([subset1, data.loc[data[attr] == j]], ignore_index = True)

            tmp1 += pi*gini(subset1[target])

        for i in range(len(items2)):
            pi = cnt2[i]/np.sum(cnt2)
            subset2 = []
            #Concat subset with attributes
            for j in items2[i]:
                if len(subset2) == 0:
                    subset2 = data.loc[data[attr] == j]
                else:
                    subset2 = pd.concat([subset2, data.loc[data[attr] == j]], ignore_index = True)

            tmp2 += pi*gini(subset2[target])

        #Select smallest gini
        gini_k = min(tmp1, tmp2)
        
            
    #No more than two attributes
    else:
        for i in range(len(items)):
            pi = cnt[i]/np.sum(cnt)
            subset = data.loc[data[attr] == items[i]]
            gini_k += pi*gini(subset[target])

    return gini_d - gini_k
    

def partitioning(data, origin, attrs, target, parent = None):
    #All samples for a given node belong to the same class
    if len(np.unique(data[target])) == 1:
        return np.unique(data[target])[0]

    #There are no remaining attributes for further partitioning - majority voting is employed for clasifying the leaf
    elif len(attrs) == 0:
        return parent
    
    #There are no samples left
    elif len(data) == 0:
        items, cnt = np.unique(origin[target], return_counts = True)
        return items[np.argmax(cnt)]

    #Partitioning process
    else:
        items, cnt = np.unique(data[target], return_counts = True)
        parent = items[np.argmax(cnt)]

        #Select information gain or gain ratio or gini index
        attr_lst = []
        for i in attrs:
            #attr_lst.append(gain(data, i, target))
            #attr_lst.append(gain_ratio(data, i, target))
            attr_lst.append(gini_index(data, i, target))

        selected_attr = attrs[np.argmax(attr_lst)]

        #Make tree node
        t = {}
        t[selected_attr] = {}
        
        #Remove selected attribute
        tmp = []
        for i in attrs:
            if i != selected_attr:
                tmp.append(i)

        #Recursive
        for i in np.unique(data[selected_attr]):
            subset = data.loc[data[selected_attr] == i]
            subtree = partitioning(subset, data, tmp, target, parent)
            t[selected_attr][i] = subtree
            
        return t

def prediction(t, attrs, mode, test):
    #Get parent node
    parent = list(t.keys())[0]
    #Get child subtree
    child = t[parent]
    #Find leaf node
    for key in child.keys():
        if test[attrs.index(parent)] == key:
            if len(child[key]) != 1:
                return child[key]
            else:
                return prediction(child[key], attrs, mode, test)
    #Not found
    #Return majoirity
    return mode

#Input data
dt_train = []
num_data = 0

f = open(sys.argv[1], 'r')

lines = f.read().split('\n')
for line in lines:
    num_data += 1
    line = line.split('\t')
    dt_train.append(line)


df_train = pd.DataFrame(dt_train)
train_header = df_train.iloc[0]
df_train = df_train[1:num_data-1]
df_train.columns = train_header

#For majority
mode = df_train.iloc[:, -1].mode()[0]

dt_test = []
num_test = 0

ff = open(sys.argv[2], 'r')

lines = ff.read().split('\n')
for line in lines:
    num_test += 1
    line = line.split('\t')
    dt_test.append(line)

df_test = pd.DataFrame(dt_test)
test_header = df_test.iloc[0]
df_test = df_test[1:num_test-1]
df_test.columns = test_header

t = partitioning(df_train, df_train, train_header[0:len(train_header)-1], train_header[len(train_header)-1])

header_lst = list(train_header)
res = []
for i in dt_test[1:-1]:
    res.append(prediction(t, header_lst, mode, i))
    

fff = open(sys.argv[3], mode='w', encoding='utf-8')

for i in range(len(header_lst)):
    fff.write(str(header_lst[i]))
    if i == len(header_lst)-1:
        fff.write('\n')
    else:
        fff.write('\t')

for i in range(1, df_test.shape[0]+1):
    for j in range(len(header_lst)):
        if j != len(header_lst)-1:
            fff.write(str(dt_test[i][j]))
            fff.write('\t')

        else:
            fff.write(str(res[i-1]))
            fff.write('\n')

    
fff.close()
