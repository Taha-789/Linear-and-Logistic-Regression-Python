import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import KFold
import math


print("----------------")
print("----------------")
print("---- PART 2 ----")
print("----------------")
print("----------------")

def predict(X, W, b):
    
    predictions = np.dot(X, W) + b
    return predictions

def derivative_W(predicted,target,X):
    size=len(target)
    loss=0
    for i in range(0,size):
            loss=loss+(X[i]*((predicted[i]-target[i])))
    return loss/size

def derivative_B(predicted,target):
    size=len(target)
    grad=0
    for i in range(0,size):
            grad=grad+((predicted[i]-target[i]))
    return grad/size


def sigmoid(z):
      return (1/(1+np.exp(-z)))


def log_loss(y_pred, target):
    eps = 1e-15
    y_pred = np.clip(y_pred, eps, 1 - eps)   # clip all probabilities safely
    loss = 0
    for i in range(len(target)):
        y = y_pred[i]
        loss += target[i]*math.log(y) + (1-target[i])*math.log(1-y)
    return -loss / len(target)


def update_W(W_old,alpha,der_W):
        return (W_old-(alpha*der_W))

def update_B(B_old,alpha,der_B):
        return (B_old-(alpha*der_B))

def convergence_check(W_old,W_new):
        check=W_new-W_old  
        norm_diff=np.linalg.norm(check)
        if norm_diff<0.0001:
            return True
        return False


df=pd.read_csv("wdbc.data")
target=df.iloc[:,1]
Y=df.iloc[:,1]
X=df.drop(df.columns[1],axis=1)
X=X.drop(df.columns[0],axis=1)

Y = Y.map({'M': 1, 'B': 0})



divide=round(len(X)*0.8)

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

X_scaled = scaler.fit_transform(X)
train_X_input = scaler.fit_transform(train_X_input)
test_X_input = scaler.transform(test_X_input)

# Initialize weights as NumPy array
W = np.zeros(train_X_input.shape[1], dtype=float)
b = 0.0


LR=[1,0.1,0.01,0.0001,0.00001,0.001] 
loss_per_ilteration=[]
max=25000

for j in range(0,6):
    W = np.zeros(train_X_input.shape[1], dtype=float)
    b = 0.0
    loss_per_ilteration=[]

    
    for i in range(0,max):
        Z = predict(train_X_input, W, b)
        y_pred=sigmoid(Z)
        loss = log_loss(y_pred, train_X_target)
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
            print("  ")
            break
    
        if convergence_check(W, W_new)==False and i==max-1:
            W = W_new
            b = b_new
            print(f"Max ilterations reached at iteration {i} with learning rate ",LR[j])
            print(f"W Value: ",W)
            print(f"B Value: ",b)
            print("  ")
            break
    
        W = W_new
        b = b_new

    plt.plot(loss_per_ilteration)
    plt.xlabel("Ilteration")
    plt.ylabel("Training loss")
    plt.title(f"GDC, Learning Rate = {LR[j]}")
    plt.show()



print("""A learning rate of 0.001 works well
       because it lets the model **learn steadily without jumping around**. A lower learning rate like 0.0001 gives slightly lower MSE
       but takes much longer to converge.
      This is why we select it !!!
""")


Z = predict(train_X_input, W, b)
y_pred=sigmoid(Z)
print(f"LOG-LOSS on Training Data is ",log_loss(y_pred, train_X_target))


y_pred = predict(test_X_input, W, b)
print(f"LOG-LOSS on Test Data is ",log_loss(y_pred, test_X_target))

Z_test = predict(test_X_input, W,b)
y_test_pred = sigmoid(Z_test)

y_test_class = (y_test_pred >= 0.5).astype(int) 

accuracy = accuracy_score(test_X_target, y_test_class)
precision = precision_score(test_X_target, y_test_class)
recall = recall_score(test_X_target, y_test_class)
f1 = f1_score(test_X_target, y_test_class)

print("Accuracy: ",accuracy)
print("Recall: ",recall)
print("Precision: ",precision)
print("F1 Score: ",f1)
cm = confusion_matrix(test_X_target, y_test_class)
print(cm)
 

print("------------")
print("---Kfold---")
print("------------")
folds=KFold(n_splits=5,shuffle=True,random_state=42)
accur=[]
precis=[]
recall=[]
f1s=[]
for train_index, test_index in folds.split(X):

    X_train = X.iloc[train_index].to_numpy()
    X_test = X.iloc[test_index].to_numpy()

    y_train = Y.iloc[train_index].to_numpy()
    y_test = Y.iloc[test_index].to_numpy()

    X_train = np.array(X_train, dtype=float)
    X_test = np.array(X_test, dtype=float)
    y_train = np.array(y_train, dtype=float)
    y_test = np.array(y_test, dtype=float)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    W = np.zeros(X_train.shape[1])
    b = 0.0

    for i in range(max):
        Z = predict(X_train, W, b)
        y_pred = sigmoid(Z)

        dw = derivative_W(y_pred, y_train, X_train)
        db = derivative_B(y_pred, y_train)

        W_new = update_W(W, 0.001, dw)
        b_new = update_B(b, 0.001, db)

        if convergence_check(W, W_new):
            W = W_new
            b = b_new
            break

        W = W_new
        b = b_new

    Z_test = predict(X_test, W, b)
    y_test_pred = sigmoid(Z_test)
    y_test_class = (y_test_pred >= 0.5).astype(int)

    accur.append(accuracy_score(y_test, y_test_class))
    precis.append(precision_score(y_test, y_test_class))
    recall.append(recall_score(y_test, y_test_class))
    f1s.append(f1_score(y_test, y_test_class))

print("Average Accuracy:", np.mean(accur))
print("Average Precision:", np.mean(precis))
print("Average Recall:", np.mean(recall))
print("Average F1:", np.mean(f1s))





print("------------")
print("---SCIKIT---")
print("------------")
model=LogisticRegression(solver='lbfgs',max_iter=25000)
model.fit(train_X_input,train_X_target)
y_pred=model.predict(test_X_input)

accuracy = accuracy_score(test_X_target, y_pred)
precision = precision_score(test_X_target, y_pred)
recall = recall_score(test_X_target, y_pred)
f1 = f1_score(test_X_target, y_pred)
    
print("Accuracy: ",accuracy)
print("Recall: ",recall)
print("Precision: ",precision)
print("F1 Score: ",f1)
cm = confusion_matrix(test_X_target, y_pred)
print(cm)
    


print("-------------")
print("----Table----")
print("-------------")
Z_test = predict(test_X_input, W, b)
y_test_pred = sigmoid(Z_test)
thresholds = [0.3, 0.5, 0.65, 0.8]

print("Threshold | Accuracy | Precision | Recall | F1-Score")
print("-----------------------------------------------")

for t in thresholds:
    y_test_class = (y_test_pred >= t).astype(int)
    
    accuracy = accuracy_score(test_X_target, y_test_class)
    precision = precision_score(test_X_target, y_test_class)
    recall = recall_score(test_X_target, y_test_class)
    f1 = f1_score(test_X_target, y_test_class)
    
    print(f"{t:<9} | {accuracy:<8.3f} | {precision:<9.3f} | {recall:<6.3f} | {f1:<8.3f}")
