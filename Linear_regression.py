import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import math



print("----------------")
print("----------------")
print("---- PART 1 ----")
print("----------------")
print("----------------")

def predict(X, W, b):
    
    predictions = np.dot(X, W) + b
    return predictions

def derivative_W(predicted,target,X):
    size=len(target)
    loss=0
    for i in range(0,size):
            loss=loss+(X[i]*((target[i]-predicted[i])))
    return -2*loss/size

def derivative_B(predicted,target):
    size=len(target)
    grad=0
    for i in range(0,size):
            grad=grad+((target[i]-predicted[i]))
    return -2*grad/size

def calculate_loss(predicted,target):
    size=len(target)
    grad=0
    for i in range(0,size):
            grad=grad+((target[i]-predicted[i])**2)
    return grad/size

def update_W(W_old,LR,der_W):
        return (W_old-(LR*der_W))

def update_B(B_old,LR,der_B):
        return (B_old-(LR*der_B))

def convergence_check(W_old,W_new):
        check=W_new-W_old  
        norm_diff=np.linalg.norm(check)
        if norm_diff<0.0001:
            return True
        return False






boston = fetch_openml(name="boston", version=1, as_frame=True)
X = boston.data   
Y = boston.target   

divide=round(len(X)*0.8)

# Convert DataFrame / Series to NumPy arrays
train_X_input = X[:divide].to_numpy()      
train_X_target = Y[:divide].to_numpy() 

test_X_input = X[divide:].to_numpy()      
test_X_target = Y[divide:].to_numpy()     

# Convert once, outside everything
train_X_input = np.array(train_X_input, dtype=float)  
train_X_target = np.array(train_X_target, dtype=float) 

test_X_input = np.array(test_X_input, dtype=float)   
test_X_target = np.array(test_X_target, dtype=float)  


scaler = StandardScaler()
train_X_input = scaler.fit_transform(train_X_input)
test_X_input = scaler.transform(test_X_input)

# Initialize weights as NumPy array
W = np.zeros(train_X_input.shape[1], dtype=float)
b = 0.0


LR=[1,0.1,0.001,0.00001,0.0001]
loss_per_ilteration=[]
max=25000

for j in range(0,5):
    W = np.zeros(train_X_input.shape[1], dtype=float)
    b = 0.0
    loss_per_ilteration=[]

    for i in range(0,max):
        y_pred = predict(train_X_input, W, b)
    
        loss = calculate_loss(y_pred, train_X_target)
        loss_per_ilteration.append(loss)
    
        dw = derivative_W(y_pred, train_X_target, train_X_input)
        db = derivative_B(y_pred, train_X_target)
    
        W_new = update_W(W, LR[j], dw)
        b_new = update_B(b, LR[j], db)
        
        if convergence_check(W, W_new):
            W = W_new
            b = b_new
            print(f"Converged at iteration {i} with learning rate ",LR[j])
            print(f"W Value: ",W)
            print(f"B Value: ",b)
            break
    
        W = W_new
        b = b_new
    print(f"MSE on Training Data is ",calculate_loss(y_pred,train_X_target))
    print("   ")

    plt.plot(loss_per_ilteration)
    plt.xlabel("Ilteration")
    plt.ylabel("Training loss")
    plt.title(f"GDC, Learning Rate = {LR[j]}")
    plt.show()

print("""A learning rate of 0.0001 works well
       because it lets the model **learn steadily without jumping around**. A lower learning rate like 0.00001 gives slightly lower MSE
       but takes much longer to converge.
      This is why we select it !!!
""")



print("-------------")
print("-----MSE-----")
print("-------------")
y_pred = predict(train_X_input, W, b)
print(f"MSE on Training Data is ",calculate_loss(y_pred,train_X_target))


y_pred = predict(test_X_input, W, b)
print(f"MSE on Test Data is ",calculate_loss(y_pred,test_X_target))


print("-------------")
print("----CHECK----")
print("-------------")
correct_count=0
for i in range(0,len(test_X_input)):
    predicted_value=np.dot(test_X_input[i], W) + b
    if abs(test_X_target[i]-round(predicted_value,3))<5:
            print(f"Correctly Identified ! Predicted: ",round(predicted_value,3)," Actual Target: ",test_X_target[i])
            correct_count+=1
    else: 
          print(f"Incorrect Identification ! Predicted: ",round(predicted_value,3)," Actual Target: ",test_X_target[i])



print("--------------")
print("---Accuracy---")
print("--------------")
print(f"Accuracy: ",correct_count/len(test_X_target))
print("   ")



print("------------")
print("---SCIKIT---")
print("------------")
lr_model = LinearRegression()
lr_model.fit(train_X_input, train_X_target)

sklearn_train_pred = lr_model.predict(train_X_input)
sklearn_test_pred = lr_model.predict(test_X_input)

train_mse_sklearn = mean_squared_error(train_X_target, sklearn_train_pred)
test_mse_sklearn = mean_squared_error(test_X_target, sklearn_test_pred)

print(f"Training MSE: {train_mse_sklearn}")
print(f"Test MSE: {test_mse_sklearn}")

correct_count_sklearn = 0
for i in range(len(test_X_input)):
    predicted_value = sklearn_test_pred[i]
    if abs(test_X_target[i] - round(predicted_value,3)) < 5:
        correct_count_sklearn += 1
accuracy_sklearn = correct_count_sklearn / len(test_X_target)
print(f"Accuracy (5 threshold, sklearn): {accuracy_sklearn}")


