'''
 Data       : All boolean inputs 
 Task       : Binary classification ( boolean output )
 Model      : Linear decision boundary, all +ve points lie above the line and -ve points
              are below (minimum flexibility )
 cost/loss  : mean squared error 
 Learning   : brute force approach to learn best parameter b
 Evaluation : Accuracy
'''

import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

class MPNeuron:
  
  def __init__(self):
    self.b = None
    
  def model(self, x):
    return(sum(x) >= self.b)
  
  def predict(self, X):
    Y = []
    for x in X:
      result = self.model(x)
      Y.append(result)
    return np.array(Y)
  
  def fit(self, X, Y):
    accuracy = {}
    
    for b in range(X.shape[1] + 1):
      self.b = b
      Y_pred = self.predict(X)
      accuracy[b] = accuracy_score(Y_pred, Y)
      
    best_b = max(accuracy, key = accuracy.get)
    self.b = best_b
    
    print('Optimal value of b is', best_b)
    print('Highest accuracy is', accuracy[best_b])


#Training

mp_neuron = MPNeuron()
mp_neuron.fit(X_binarised_train, Y_train)

#Prediction

Y_test_pred = mp_neuron.predict(X_binarised_test)
accuracy_test = accuracy_score(Y_test_pred, Y_test)