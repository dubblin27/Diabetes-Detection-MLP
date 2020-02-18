from scipy.io import arff
import mlp
import numpy as np
from sklearn.model_selection import train_test_split 

data = np.array(arff.loadarff('1year.arff')[0])
np.random.shuffle(data)
output = [int(list(i).pop()) for i in data]
input = [list(i)[:-1] for i in data]
ratio = 0.8
length = int(len(input) * ratio)
# X_train = np.array(input[:length])

# y_train = np.array(output[:length])
# y_test = np.array(output[length:])
# X_test = np.array(input[length:])
X_train,X_test, y_train,y_test = train_test_split(input,output,test_size = 0.8,random_state=101)

machine = mlp.mlp(inputs=X_train, targets=y_train, nhidden=4, beta=.2, momentum=0.5, outtype='logistic')
machine.mlptrain(inputs=X_train, targets=y_train, eta=0.2, niterations=100)
print(machine.output(X_test))