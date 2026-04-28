import numpy as np
from data.preprocessing import get_data


X,y= get_data()
#Initializing parameters
def parameters(X_n,y_n,h_n):
    np.random.seed(42)
    
    W1= np.random.randn(h_n,X_n)
    b1= np.zeros((h_n,1))
    
    W2= np.random.randn(y_n,h_n)
    b2= np.zeros((y_n,1))
    
    return W1,b1,W2,b2

#Activation Functions
def relu(Z):
    return np.maximum(Z,0)

def sigmoid(Z):
    return 1/(1+ np.exp(-Z))


#Forward Pass (Formula used:- Z= W.X + b)
def forward_pass(X,W1,b1,W2,b2):
    
    #Current shapes
    #print("W1:", W1.shape)
    #print("X:", X.shape)
    #print("W2:", W2.shape)
    
    
    Z1= np.dot(W1,X)+b1
    A1= relu(Z1)
    
    Z2= np.dot(W2,A1)+b2
    A2= sigmoid(Z2)

    return Z1,A1,Z2,A2

#Loss Calculation
def Loss(A2,y):
    m= y.shape[1]
    
    #prevent undefined log(0)
    A2 = np.clip(A2, 1e-8, 1 - 1e-8)
    
    loss = - (1/m) * np.sum(y*np.log(A2) + (1-y)*np.log(1-A2))
    
    return loss

X_n= X.shape[0]
h_n= 10
y_n=1

W1,b1,W2,b2= parameters(X_n,y_n,h_n)
Z1,A1,Z2,A2= forward_pass(X,W1,b1,W2,b2)

loss= Loss(A2,y)

print("Loss= ",loss)

#Backpropogation

def backprop(X,y,Z1,A1,Z2,A2,W2):
    m = X.shape[1]
    
    dZ2 = A2 - y
    dW2 = (1/m) * np.dot(dZ2, A1.T)
    db2 = (1/m) * np.sum(dZ2, axis=1, keepdims=True)
    
    dZ1 = np.dot(W2.T, dZ2) * (Z1 > 0)
    dW1 = (1/m) * np.dot(dZ1, X.T)
    db1 = (1/m) * np.sum(dZ1, axis=1, keepdims=True)
    
    return dW1, db1, dW2, db2

def update(W1,b1,W2,b2,dW1,db1,dW2,db2,lr):
    W1 -= lr * dW1
    b1 -= lr * db1
    W2 -= lr * dW2
    b2 -= lr * db2
    
    return W1, b1, W2, b2

epochs = 1000
lr = 0.01

for i in range(epochs):
    Z1, A1, Z2, A2 = forward_pass(X, W1, b1, W2, b2)
    
    loss = Loss(A2, y)
    
    dW1, db1, dW2, db2 = backprop(X, y, Z1, A1, Z2, A2, W2)
    
    W1, b1, W2, b2 = update(W1, b1, W2, b2, dW1, db1, dW2, db2, lr)
    
    if i % 100 == 0:
        print("Loss:", loss)


preds = (A2 > 0.5).astype(int)
accuracy = np.mean(preds == y)
print("Accuracy percentage:", accuracy*100)