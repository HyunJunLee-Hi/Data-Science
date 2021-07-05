import sys
import numpy as np
import pandas as pd

#Matrix factorization
def MF(rating_matrix, U, V):
    for i in range(50):
        for j in range(users_num + 1):
            for k in range(len(rating_matrix[i])):
                #If item has rating
                if rating_matrix[j][k] > 0:
                    #Calculate local minimum error
                    err = rating_matrix[j][k] - np.dot(U[j][0], V[0][k])
                    U[j][0] = U[j][0] + 0.002 * (2 * err * V[0][k])
                    V[0][k] = V[0][k] + 0.002 * (2 * err * U[j][0])

        #Check error for break       
        e = 0
        for i in range(users_num + 1):
            for j in range(len(rating_matrix[i])):
                if rating_matrix[i][j] > 0:
                    e += (rating_matrix[i][j] - np.dot(U[i][0], V[0][j]))**2

        if e < 60000.:
            break
        
    return np.dot(U, V)

train_data = []
test_data = []

#Input training data
f = open(sys.argv[1], 'r')
train_lines = f.read().split('\n')
tmp = None
for line in train_lines:
    if line != '':
        line = line.split('\t')
        tmp = list(map(int, line))
        train_data.append(tmp)

#Input test data
ff = open(sys.argv[2], 'r')
test_lines = ff.read().split('\n')
tmp = None
for line in test_lines:
    if line != '':
        line = line.split('\t')
        tmp = list(map(int, line))
        test_data.append(tmp)

train_data = np.array(train_data, dtype=int)
train_data[:,0] -= 1
train_data[:,1] -= 1
test_data = np.array(test_data, dtype=int)
test_data[:,0] -= 1
test_data[:,1] -= 1

#Make (user * item) rating matrix
users_num = np.max(train_data[:,0])
items_num = np.max(train_data[:,1])
shape = (users_num + 1, items_num + 1)

rating_matrix = np.ndarray(shape, dtype=float)

for user, item, rating, time in train_data:
    rating_matrix[user][item] = rating

#Initial U, V
U = np.random.rand(users_num + 1, 1)
V = np.random.rand(1, items_num + 1)

#Compute matrix factorization
res = MF(rating_matrix, U, V)

#Output txt file
fff = open(sys.argv[1] + '_prediction.txt', mode='w', encoding='utf-8')
for i in range(len(test_data)):
    if i != '':
        try:
            fff.write(str(test_data[i][0]+1))
            fff.write('\t')
            fff.write(str(test_data[i][1]+1))
            fff.write('\t')
            fff.write(str(float(res[test_data[i][0]][test_data[i][1]])))
            fff.write('\n')
        except:
            fff.write(str(0))
            fff.write('\n')
fff.close()
