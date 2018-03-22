import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.special import expit

##################
# Importing data #
##################

X_train = np.genfromtxt('X_train.csv', delimiter=",")
y_train = np.genfromtxt('y_train.csv', delimiter=",")
X_test = np.genfromtxt('X_test.csv', delimiter=",")
y_test = np.genfromtxt('y_test.csv', delimiter=",")

####################
# Helper Functions #
####################

def createBinaryConfusionMatrix(predictions, ground_truth):
    conf = np.zeros((2,2)).astype(np.int)
    for i in range(ground_truth.shape[0]):
        a = int(ground_truth[i])
        b = int(predictions[i])
        conf[a,b] += 1
    return conf

def calculateAccuracy(predictions, ground_truth):
    n = predictions.shape[0]
    assert n == ground_truth.shape[0]
    incorrect = np.sum(np.logical_xor(predictions, ground_truth))
    return (1 - incorrect/n)

###################################
# Problem 2(a) - BAYES CLASSIFIER #
###################################
        
class BayesClassifier:
    def __init__(self):
        self.theta = None
        self.pi = None
        self.num_features = None
    
    def train(self, X, y):
        n, D = X.shape
        assert n == y.shape[0]
        theta = np.zeros((2, D))
        self.pi = np.mean(y)
        X_0 = X_train[y_train==0, ]
        X_1 = X_train[y_train==1, ]
        theta[0,:54] = np.mean(X_0[:,0:54], axis=0)
        theta[0,54:] = 1/np.mean(np.log(X_0[:,54:]), axis=0)
        theta[1,:54] = np.mean(X_1[:,0:54], axis=0)
        theta[1,54:] = 1/np.mean(np.log(X_1[:,54:]), axis=0)
        self.theta = theta
        self.num_features = D
        
    def predict(self, x):
        pi = self.pi
        theta = self.theta
        p =np.zeros(2)
        for i in range(2):
            prod = (pi**(i))*(1-pi)**(1-i)
            for d in range(54):
                prod *= (((theta[i,d])**(x[d]))*((1-theta[i,d])**(1-x[d])))
            for d in range(54, 57):
                prod *= (theta[i,d]*(x[d]**(-(theta[i,d] + 1))))
            p[i] = prod
        return np.argmax(p)
                
    def predictBatch(self, X):
        results = []
        for n in range(X.shape[0]):
            results.append(self.predict(X[n]))
        return np.array(results)

B = BayesClassifier()
B.train(X_train, y_train)

results = B.predictBatch(X_test)

conf = createBinaryConfusionMatrix(results, y_test)

plt.figure(figsize=(5, 5))
sns.heatmap(conf, annot=True, annot_kws={"size": 20})
plt.xlabel('Predictions')
plt.ylabel('Ground Truth')
plt.savefig('2a.png')

bayes_acc = calculateAccuracy(results, y_test)
print('Bayes Classifier Test Accuracy: {:.2f}%'.format(bayes_acc*100))

#########################################
# Problem 2(b) - STEM PLOTS + INFERENCE #
#########################################

fig = plt.figure(figsize=(20, 10))
sns.set()
plt.stem(range(1, 55), B.theta[0, :54], )
plt.xlabel('Features', fontsize=20)
plt.ylabel('Parameters for Y = 0', fontsize=20)
plt.xlim(0, 55)
plt.tick_params(axis='both', labelsize=20)
plt.savefig('2b1.png')

fig = plt.figure(figsize=(20,10))
sns.set()
plt.stem(range(1, 55), B.theta[1, :54], )
plt.xlabel('Features', fontsize=20)
plt.ylabel('Parameters for Y = 1', fontsize=20)
plt.xlim(0, 55)
plt.tick_params(axis='both', labelsize=20)
plt.savefig('2b2.png')

#################################
# Problem 2(c) - KNN Classifier #
#################################

class kNN():
    def __init__(self, X, y):
        self.X = X
        self.y = y
        
    def distance(self, x1, x2):
        return np.linalg.norm(x1 - x2, ord=1)
        
    def predict(self, x, k = 1):
        X = self.X
        y = self.y
        d = np.zeros(X.shape[0])
        temp = np.column_stack((d, y))
        for i in range(X.shape[0]):
            temp[i, 0] = self.distance(x, X[i,:])
        temp = temp[temp[:,0].argsort()]
        return int(np.mean(temp[:k, 1]) > 0.5)
    
    def predictForMultipleK(self, x, k = [1]):
        X = self.X
        y = self.y
        d = np.zeros(X.shape[0])
        temp = np.column_stack((d, y))
        for i in range(X.shape[0]):
            temp[i, 0] = self.distance(x, X[i,:])
        temp = temp[temp[:,0].argsort()]
        return np.array([int(np.mean(temp[:i, 1]) > 0.5) for i in k])
    
    def predictBatch(self, x, k = 1):
        kresult = []
        for i in range(x.shape[0]):
            kresult.append(self.predict(x[i], k))
        return np.array(kresult)
    
    def predictBatchForMultipleK(self, x, k = [1]):
        kresult = []
        for i in range(x.shape[0]):
            kresult.append(self.predictForMultipleK(x[i], k))
        return np.array(kresult)

k = kNN(X_train, y_train)

knn_pred_till20 = k.predictBatchForMultipleK(X_test, range(1, 21))

acc_vs_k = [calculateAccuracy(knn_pred_till20[:, i-1], y_test) for i in range(1, 21)]

fig = plt.figure(figsize=(10, 10))
sns.set_style('whitegrid')
plt.plot(acc_vs_k)
plt.xlabel('K - # of Nearest Neigbours', fontsize=20)
plt.ylabel('Accuracy', fontsize=20)
plt.tick_params(axis='both', labelsize=20)
plt.savefig('2c.png')

######################################
# Problem 2(d) - Logistic Regression #
######################################

y_lr = np.array([1 if i == 1 else -1 for i in y_train])
X_lr = np.column_stack((np.ones(X_train.shape[0]), X_train))

class LogRegression():
    def __init__(self):
        self.eta = None
        self.X = None
        self.y = None
        self.w = None
    
    def calcGrad(self, w):
        X = self.X
        y = self.y
        (n, d) = X.shape
        Xw = X.reshape(n, d).dot(w.reshape(d, 1))
        z = np.multiply(Xw, y.reshape(n, 1))
        val = expit(z)
        assert val.shape == (n, 1)
        y_temp = np.multiply(y.reshape(n, 1), (1.- val).reshape(n, 1))
        y_temp2 = (y_temp.T*X.T).T
        s = np.sum(y_temp2, axis=0, keepdims=True).T
        assert s.shape == (d, 1)
        L = np.sum(np.log(val), keepdims=True)
        return s, L
           
    def fit(self, X, y, steps):
        self.X = X
        self.y = y
        n, d = X.shape[0], X.shape[1]
        w = np.zeros((d,1), dtype=float)
        Loss = np.zeros(steps)
        for t in range(steps):
            eta = 1/(100000*((t+2)**0.5))
            s, L = self.calcGrad(w)
            w = w + eta*(s)
            Loss[t] = L
        self.w = w
        self.L = Loss
        
    def predictBatch(self, X):
        w = self.w
        n, d = X.shape[0], X.shape[1]
        z = X.dot(w)
        assert z.shape == (n, 1)
        return expit(z)

lR = LogRegression()

lR.fit(X_lr, y_lr, 10000)

fig = plt.figure(figsize=(20, 10))
plt.plot(lR.L[1:])
plt.xlabel('Iterations', fontsize=20)
plt.ylabel('Objective function', fontsize=20)
plt.savefig('2d.png')

X_test_lr = np.column_stack((np.ones(X_test.shape[0]), X_test))
res = lR.predictBatch(X_test_lr)
res[res>0.5] = 1
res[res<=0.5] = 0
res = res.reshape(-1)

lR_acc = calculateAccuracy(res, y_test)
print('Logistic Regression Classifier Test Accuracy: {:.2f}%'.format(lR_acc*100))

############################################################
# Problem 2(e) - Logistic Regression using Newton's Method #
############################################################

class LogRegressionNewtons():
    def __init__(self):
        self.X = None
        self.y = None
        self.w = None
    
    def calcHessian(self, w):
        X = self.X
        y = self.y
        (n, d) = X.shape
        Xw = X.reshape(n, d).dot(w.reshape(d, 1))
        val = expit(Xw)
        assert val.shape == (n, 1)
        s = np.zeros((d, d))
        for i in range(n):
            xi = X[i, :].reshape(d, 1)
            xixiT = np.dot(xi, xi.T)
            assert xixiT.shape == (d, d)
            v = float(val[i])
            si = (-1)*(v*(1.- v))*xixiT
            assert si.shape == (d, d)
            s = s + si
            assert s.shape == (d, d)
        return s
    
    def calcGrad(self, w):
        X = self.X
        y = self.y
        (n, d) = X.shape
        Xw = X.reshape(n, d).dot(w.reshape(d, 1))
        z = np.multiply(Xw, y.reshape(n, 1))
        val = expit(z)
        assert val.shape == (n, 1)
        y_temp = np.multiply(y.reshape(n, 1), (1.- val).reshape(n, 1))
        y_temp2 = (y_temp.T*X.T).T
        s = np.sum(y_temp2, axis=0, keepdims=True).T
        assert s.shape == (d, 1)
        L = np.sum(np.log(val), keepdims=True)
        return s, L

            
    def fit(self, X, y, steps):
        self.X = X
        self.y = y.reshape(y.shape[0], 1)
        n, d = X.shape[0], X.shape[1]
        w = np.zeros((d,1))
        Loss = np.zeros(steps)
        for t in range(steps):
            eta = 1/(100000*((t+2)**0.5))
            s, L = self.calcGrad(w)
            h = self.calcHessian(w)
            assert h.shape == (d, d)
            hinv = np.linalg.inv(h)
            assert hinv.shape == (d, d)
            w = w - eta*(hinv.dot(s))
            Loss[t] = L
        self.w = w
        self.L = Loss
        
    def predictBatch(self, X):
        w = self.w
        (n, d) = X.shape
        assert w.shape == (d, 1)
        z = X.dot(w).reshape(n, 1)
        return expit(z)

X_lr = X_lr.astype(np.float64)
y_lr = y_lr.astype(np.float64)
lRN = LogRegressionNewtons()
lRN.fit(X_lr, y_lr, 100)

fig = plt.figure(figsize=(20, 10))
plt.plot(lRN.L[:])
plt.xlabel('Iterations', fontsize=20)
plt.ylabel('Objective function', fontsize=20)
plt.savefig('2e.png')

res = lRN.predictBatch(X_test_lr)
res[res>0.5] = 1
res[res<=0.5] = 0
res = np.array(res).reshape(-1)

lRN_acc = calculateAccuracy(res, y_test)
print('Logistic Regression Classifier (Modified) Test Accuracy: {:.2f}%'.format(lRN_acc*100))