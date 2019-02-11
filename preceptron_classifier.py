"""
clssification problem, back propagation using Rosenblatt's percepteron
source: https://sebastianraschka.com/Articles/2015_singlelayer_neurons.html
graph: [0,0] [0,0] [0]
each input layer is connected to all the hidden layers
call inp layer: a1,a2
call hidden layer: l1,l2
call output layer: o
call weights of 1st layer = w1,w2,w3,w4
call weights of output layer = v1,v2
alg:
l1 = sigmoid(a1*w1+a2*w3)
l2 = sigmoid(a1*w2+a2*w4)
o = sigmoid(v1*l1+v2*l2)
learning alg:
initialize weights to 0 at the begning of the run
weight +=learning rate*error*input vector
"""
#initial imports
import math
import matplotlib.pyplot as plt

def evaluate_nn_l1(inpvec,weightvec,hidden_layer,lno):
    if lno==1:
        print('input vec l1:',inpvec)
        print('weights l1:',weightvec)
        print('evaluating 1st layer')
        for i in range(2):
            for j in range(2):
                hidden_layer[i] += inpvec[j]*weightvec[i][j]
        hidden_layer = [sigmoid(i) for i in hidden_layer]
        print('hidden layer:',hidden_layer)
        return hidden_layer

def evaluate_nn_l2(inpvec,weightvec,output,lno):
    if lno == 2:
        print('input vec l2:',inpvec)
        print('weights l2:',weightvec)
        print('evaluating output layer')
        for i in range(2):
            output += weightvec[i]*inpvec[i]
        output = sigmoid(output)
        print('output:',output)
        return output

def evaluate_nn_error_l2(desired,actual,lno):
    if lno == 2:
        print('error_l2:',desired - actual)
    return (desired - actual)

def evaluate_nn_error_l1(inpvec,hidden_layer,lno):
    err = []
    if lno ==1:
        for i,val in enumerate(inpvec):
            err.append(val-hidden_layer[i])
        print('error_l1:',err)
    return err

def learn_nn_l2(inpvec,weightvec,error,learning_Rate,lno):
    if lno==2:
        print('learning at l2')
        for i in range(2):
            weightvec[i] += learning_Rate*error*inpvec[i]
        print('updated weights_l2:',weightvec)
        return weightvec

def learn_nn_l1(inpvec,weightvec,error,learning_Rate,lno):
    if lno==1:
        print('learning at l1')
        for i in range(2):
            for j in range(2):
                weightvec[i][j] += learning_Rate*error[j]*inpvec[i]
        print('updated weights_l1:',weightvec)
        return weightvec

def sigmoid(x):
    return 1/(1+math.exp(-x))

inputs = [1,1]
hidden_layer = [0,0]
weights_l1 = [[0,0],[0,0]]
weights_l2 = [0,0]
desired_result = 1
learning_Rate = 0.2
trials = 100
result = 0
error_l1_list = []
error_l2_list = []
# trainer run1
for i in range(trials):
    print('trial:',i)
    hid_layer = evaluate_nn_l1(inputs,weights_l1,hidden_layer,lno = 1)
    res = evaluate_nn_l2(hid_layer,weights_l2,result,lno = 2)
    err_l2 = evaluate_nn_error_l2(desired_result,res,lno=2)
    err_l1 = evaluate_nn_error_l1(inputs,hid_layer,lno=1)
    wv_l2 = learn_nn_l2(hid_layer,weights_l2,err_l2,learning_Rate,lno=2)
    wv_l1 = learn_nn_l1(inputs,weights_l1,err_l1,learning_Rate,lno=1)
    weights_l1 = wv_l1
    weights_l2 = wv_l2
    error_l1_list.append(err_l1)
    error_l2_list.append(err_l2)
    print('\n')

# plotting
# plt.plot([i for i in range(trials)],error_l1_list)
# plt.plot([i for i in range(trials)],error_l2_list)
# plt.show()
