#Jonas Gabirot

import numpy as np

import pandas as pd
from sklearn.model_selection import train_test_split


df_result = pd.read_csv('train_result.csv')

df = pd.read_csv('train.csv')

df_result = df_result.drop(columns=["Index"])

result_train, result_valid = train_test_split(df_result, test_size=0.2, shuffle=False)
train_X, valid = train_test_split(df, test_size=0.2, shuffle=False)

result_train = result_train.to_numpy()
result_valid = result_valid.to_numpy()
train_X = train_X.to_numpy()
valid = valid.to_numpy()

train_X = np.delete(train_X, 1568 ,axis=1)
valid = np.delete(valid, 1568 ,axis=1)

print(train_X.shape)



def softmax(x):
    e_x = np.exp(x)
    return np.divide(e_x, np.sum(e_x))

def train(X, Y, epochs, lr):
    
    # X --> Input.
    # y --> true/target value.
    # epochs --> Number of iterations.
    # lr --> Learning rate.
        
    # m-> number of training examples
    # n-> number of features 
    m, n = X.shape
    
    classes = np.unique(Y).size
    
    # Initializing weights and biases.
    w = np.random.rand(n,classes)
    b = np.zeros(classes)
    
    
    
    # Training loop.
    for epoch in range(epochs):
      for x,y in zip(X,Y):
        t = np.zeros(classes)    
        t[y] += 1
         
        g = softmax(np.dot(w.T, x) + b)
        
        
        x = x[...,None]
        new_g = g-t
        new_g = new_g[...,None]
        
        dw = np.dot((new_g),x.T)
        db = g-t
        
         
        # Calcul gradient
        # Updating the parameters.
        w -= lr*dw.T
        b -= lr*db
        
        
        
      print(epoch)  
    # returning weights, bias
    return w, b



def predict(X, w, b):
    
    # X --> Input.
    preds = []
    # Calculating predictions/y_hat.
    for x in X:
      pred =  np.argmax(softmax(np.dot(w.T, x) + b))
      preds.append(pred)
    
    
    return preds

w, b= train(train_X, result_train, epochs=15, lr=0.05)

preds = predict(valid, w, b)

def accuracy(preds, true_labels):
    n_correct = 0
    n_samples = 0
    for i in range(len(preds)):
          label = true_labels[i]
          pred = preds[i]
          if (label == pred):
              n_correct += 1
          n_samples += 1
    
    acc = 100.0 * n_correct / n_samples
    return acc

print(accuracy(preds,result_valid))

