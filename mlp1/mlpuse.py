from scipy.io import arff
import mlp
import numpy as np
data = np.array(arff.loadarff('1year.arff')[0])
np.random.shuffle(data)
output = [int(list(i).pop()) for i in data]
input = [list(i)[:-1] for i in data]
ratio = 0.8
length = int(len(input) * ratio)
trainInput = np.array(input[:length])
trainOutput = np.array(output[:length])
testOutput = np.array(output[length:])
testInput = np.array(input[length:])
machine = mlp.mlp(inputs=trainInput, targets=trainOutput, nhidden=4, beta=.2, momentum=0.5, outtype='logistic')
machine.mlptrain(inputs=trainInput, targets=trainOutput, eta=0.2, niterations=100)
print(machine.output(testInput))