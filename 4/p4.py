import numpy as np
x = np.array(([2,9], [1,5],[3,6]), dtype = float)
y = np.array(([92], [86], [89]), dtype=float)
x = x/np.amax(x,axis=0)
y = y/100

def sigmoid(x):
	return (1/(1+np.exp(-x)))

def derivatives_sigmoid(x):
	return x*(1-x)

#initialize network 
epoch = 20000
lr = 0.1
inputlayer = 2
hiddenlayer = 3
output = 1

wh = np.random.uniform(size=(inputlayer, hiddenlayer))
bh = np.random.uniform(size=(1, hiddenlayer))
wout = np.random.uniform(size=(hiddenlayer,output))
bout = np.random.uniform(size=(1, output))

for i in range(epoch):
	hinp1 = np.dot(x,wh)
	hinp = hinp1 + bh
	hlayer_act = sigmoid(hinp)
	outinp1 = np.dot(hlayer_act, wout)
	outinp = outinp1 + bout
	output = sigmoid(outinp)

EO = y-output
outgrad = derivatives_sigmoid(output)
d_output = EO*outgrad

Eh = d_output.dot(wout.T)
hiddengrad = derivatives_sigmoid(hlayer_act)

d_hiddenlayer = Eh * hiddengrad
wout += hlayer_act.T.dot(d_output) * lr

bout += np.sum(d_output, axis=0, keepdims=True)*lr
wh += x.T.dot(d_hiddenlayer) * lr

print("Input: \n" + str(x))
print("Actual Output: \n" + str(y))
print("Predicted Output: \n", output)
