"""
simple fully connected ann with 0 weights and 1 hidden layer.
each input is connected to only one hidden node.
this network ios 100% overfitting
the weight vector notation used here is actually not weights per se, but the nodes of the
hidden layer
graph:
[[0,0,0,0],[0,0,0,0],[0]]
source: https://becominghuman.ai/making-a-simple-neural-network-2ea1de81ec20
"""
def evaluate_nn(inpvec,weightvec):
    result = 0
    for i,val in enumerate(inpvec):
        layer_val = val*weightvec[i]
        result += layer_val
    return round(result,2)

def evaluate_nn_error(desired,actual):
    return (desired - actual)

def learn_nn(inpvec,weightvec,learning_Rate):
    for i,val in enumerate(weightvec):
        if inpvec[i]>0:
            weightvec[i] += learning_Rate
    return inpvec, weightvec

def trainer(trials,inpv,weightv,learning_Rate,desired):
    for i in range(trials):
        print('trial:',i)
        print('inpv:',inpv)
        print('weightv:',weightv)
        print('learning rate:',learning_Rate)
        res = evaluate_nn(inpv,weightv)
        inpv1, weightv1 = learn_nn(inpv,weightv,learning_Rate)
        error = evaluate_nn_error(desired,res)
        print('error:',error)
        inpv = inpv1
        weightv = weightv1
        print('\n')
        if error ==0:
            break

#init
inputs = [0,1,0,0] # the seconprint('learning rate:',learning_Rate)d button is pressed
weights = [0,0,0,0]
desired_result = 1
learning_Rate = 0.20
trials = 10

# evaluate_nn 1st run without learning
# res = evaluate_nn(inputs,weights)
# print('result:',res)
# err = evaluate_nn_error(desired_result,res)
# print('error:',err)

# evaluate_nn 2st run with learning
# inpv,weightv = learn_nn(inputs,weights,learning_Rate)
# res = evaluate_nn(inputs,weights)
# err = evaluate_nn_error(desired_result,res)
# print('inpv:',inpv)
# print('weightv:',weightv)
# print('result:',res)
# print('error:',err)

#evaluate_nn with trainer function
trainer(trials,inputs,weights,learning_Rate,desired_result)
