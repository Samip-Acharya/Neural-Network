import numpy as np
# import matplotlib.pyplot as plt

class Perceptron:
    def __init__(self):
        self.step = lambda x : 1 if x>=0 else 0

    def train(self,X,y): 
        epochs = 100
        X = np.insert(X,0,1,axis=1)
        self.wt = np.zeros(X.shape[1])
        lr = 0.1
        for _ in range(epochs):
            i = np.random.randint(0,X.shape[0])
            y_i = y[i]
            y_hat = self.step(np.dot(X[i],self.wt))
            self.wt = self.wt + lr*(y_i-y_hat)*X[i]
            
    def predict(self,x_input):
        x = np.array(x_input)
        x = np.insert(x,0,1)
        return self.step(np.dot(x,self.wt))


X = np.array([[0,0],[0,1],[1,0],[1,1]])
y= np.array([0,0,0,1])

model = Perceptron()
model.train(X,y)
for x in X:
    y = model.predict(x)
    print(x,end="-> ")
    print(y)
