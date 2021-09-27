from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import fetch_mldata

## task1
### load data
iris_dataset = load_iris()
X_train, X_test, Y_train, Y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], random_state = 0)

### Decision Tree
clf = DecisionTreeClassifier(criterion='gini',random_state= 0)
clf.fit(X_train, Y_train)
print(clf)
predicted = clf.predict(X_train)
print(predicted)
print('train data accuracy:', clf.score(X_train, Y_train))
print('test data accuracy:', clf.score(X_test, Y_test))

clf = DecisionTreeClassifier(criterion='gini', max_depth = 3, random_state= 0)
clf.fit(X_train, Y_train)
print(clf)
predicted = clf.predict(X_train)
print(predicted)
print('train data accuracy:', clf.score(X_train, Y_train))
print('test data accuracy:', clf.score(X_test, Y_test))

clf = DecisionTreeClassifier(criterion='entropy', random_state= 0)
clf.fit(X_train, Y_train)
print(clf)
predicted = clf.predict(X_train)
print(predicted)
print('train data accuracy:', clf.score(X_train, Y_train))
print('test data accuracy:', clf.score(X_test, Y_test))

clf = DecisionTreeClassifier(criterion='entropy', max_depth = 3, random_state= 0)
clf.fit(X_train, Y_train)
print(clf)
predicted = clf.predict(X_train)
print(predicted)
print('train data accuracy:', clf.score(X_train, Y_train))
print('test data accuracy:', clf.score(X_test, Y_test))

num = len(X_train)
data_num = []
train_acc = []
test_acc= []
for i in range(10):
    clf = DecisionTreeClassifier(max_depth = 10, random_state= 0)
    data_num.append(int(num/10*(i+1)))
    x_train = X_train[:int(num/10*(i+1))]
    y_train = Y_train[:int(num/10*(i+1))]
    clf.fit(x_train, y_train)
    train_acc.append(clf.score(x_train, y_train))
    test_acc.append(clf.score(X_test, Y_test))
    print('# of training instance:', int(num/10*(i+1)))
    print('train data accuracy:', clf.score(x_train, y_train))
    print('test data accuracy:', clf.score(X_test, Y_test))
    
plt.plot(data_num,train_acc,color='red',label='train_acc')
plt.plot(data_num,test_acc,color='blue',label='test_acc')

plt.title(u'learning curve')
plt.xlabel(u'size of training data')
plt.ylabel(u'accuracy')

plt.legend()

num = len(X_train)
max_depth = []
train_acc = []
test_acc= []
for i in range(15):
    clf = DecisionTreeClassifier(max_depth = i+1, random_state= 0)
    max_depth.append(i+1)
    x_train = X_train
    y_train = Y_train
    clf.fit(x_train, y_train)
    train_acc.append(clf.score(x_train, y_train))
    test_acc.append(clf.score(X_test, Y_test))
    print('max_depth:', i+1)
    print('train data accuracy:', clf.score(X_train, Y_train))
    print('test data accuracy:', clf.score(X_test, Y_test))
    
plt.plot(max_depth,train_acc,color='red',label='train_acc')
plt.plot(max_depth,test_acc,color='blue',label='test_acc')

plt.title(u'max_depth VS accuracy')
plt.xlabel(u'max_depth')
plt.ylabel(u'accuracy')

plt.legend()

### Neural Network
NN_model = MLPClassifier(hidden_layer_sizes = (100), random_state = 0, max_iter = 1000).fit(X_train, Y_train)
print('train data accuracy:', NN_model.score(X_train, Y_train))
print('test data accuracy:', NN_model.score(X_test, Y_test))

num = len(X_train)
iter_num = []
train_acc = []
test_acc= []
for i in range(10):
    NN_model = MLPClassifier(hidden_layer_sizes = (100), random_state = 0, max_iter = 100*(i+1)).fit(X_train, Y_train)
    iter_num.append(100*(i+1))
    x_train = X_train
    y_train = Y_train
    NN_model.fit(x_train, y_train)
    train_acc.append(NN_model.score(x_train, y_train))
    test_acc.append(NN_model.score(X_test, Y_test))
    print('# of iter:', 100*(i+1))
    print('train data accuracy:', NN_model.score(x_train, y_train))
    print('test data accuracy:', NN_model.score(X_test, Y_test))
    
plt.plot(iter_num,train_acc,color='red',label='train_acc')
plt.plot(iter_num,test_acc,color='blue',label='test_acc')

plt.title(u'learning curve')
plt.xlabel(u'# of iterations')
plt.ylabel(u'accuracy')

plt.legend()

NN_model = MLPClassifier(hidden_layer_sizes = (100,100), random_state = 0, max_iter = 200).fit(X_train, Y_train)
plt.legend()print('train data accuracy:', NN_model.score(X_train, Y_train))
print('test data accuracy:', NN_model.score(X_test, Y_test))

NN_model = MLPClassifier(hidden_layer_sizes = (5,5,5), random_state = 0, max_iter = 100000).fit(X_train, Y_train)
print('train data accuracy:', NN_model.score(X_train, Y_train))
print('test data accuracy:', NN_model.score(X_test, Y_test))

### AdaBoost Tree
adbot = AdaBoostClassifier(DecisionTreeClassifier(criterion='gini',random_state= 0), random_state = 0)
adbot.fit(X_train,Y_train)
print('train data accuracy:', adbot.score(X_train, Y_train))
print('test data accuracy:', adbot.score(X_test, Y_test))

num = len(X_train)
data_num = []
train_acc = []
test_acc= []
for i in range(10):
    adbot = AdaBoostClassifier(DecisionTreeClassifier(criterion='gini',random_state= 0), random_state = 0)  
    data_num.append(int(num/10*(i+1)))
    x_train = X_train[:int(num/10*(i+1))]
    y_train = Y_train[:int(num/10*(i+1))]
    adbot.fit(x_train, y_train)
    train_acc.append(adbot.score(x_train, y_train))
    test_acc.append(adbot.score(X_test, Y_test))
    print('# of training instance:', int(num/10*(i+1)))
    print('train data accuracy:', adbot.score(x_train, y_train))
    print('test data accuracy:', adbot.score(X_test, Y_test))
    
plt.plot(data_num,train_acc,color='red',label='train_acc')
plt.plot(data_num,test_acc,color='blue',label='test_acc')

plt.title(u'learning curve')
plt.xlabel(u'size of training data')
plt.ylabel(u'accuracy')

plt.legend()

### SVM
svm_model = svm.SVC(C=1.0, kernel = 'rbf', decision_function_shape = 'ovr', gamma = 0.01, random_state = 0)
svm_model.fit(X_train, Y_train)
print('train data accuracy:', svm_model.score(X_train, Y_train))
print('test data accuracy:', svm_model.score(X_test, Y_test))

svm_model = svm.SVC(C=1.0, kernel = 'linear', decision_function_shape = 'ovr', gamma = 0.01, random_state = 0)
svm_model.fit(X_train, Y_train)
print('train data accuracy:', svm_model.score(X_train, Y_train))
print('test data accuracy:', svm_model.score(X_test, Y_test))

svm_model = svm.SVC(C=1.0, kernel = 'sigmoid', decision_function_shape = 'ovr', gamma = 0.01, random_state = 0)
svm_model.fit(X_train, Y_train)
print('train data accuracy:', svm_model.score(X_train, Y_train))
print('test data accuracy:', svm_model.score(X_test, Y_test))

### KNN
knn_model = KNeighborsClassifier(3)
knn_model.fit(X_train, Y_train)
print('train data accuracy:', knn_model.score(X_train, Y_train))
print('test data accuracy:', knn_model.score(X_test, Y_test))


## task2
### load data
mnist = fetch_mldata('MNIST original', data_home = './datasets')
X, Y = mnist['data'], mnist['target'] 
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state = 0)