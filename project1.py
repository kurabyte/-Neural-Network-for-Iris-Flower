import numpy as np
from sklearn.datasets import load_iris

#1 load data
iris = load_iris()
X = iris.data
Y = iris.target

#2 shuffle the data
np.random.seed(42)
indices = np.random.permutation(len(X))
X_shuffled = X[indices]
Y_shuffled = Y[indices]

#3 split train(80%) test(20%)
split_index = int(0.8 * len(X))
X_train,X_test = X_shuffled[:split_index], X_shuffled[split_index:]
Y_train,Y_test = Y_shuffled[:split_index], Y_shuffled[split_index:]

print(f"Training data shape:{X_train.shape}")
print(f"Testing data shape:{X_test.shape}")

#4 standardization
X_train_mean = np.mean(X_train,axis=0)
X_train_std = np.std(X_train,axis=0)

X_train_norm = (X_train-X_train_mean)/X_train_std
X_test_norm = (X_test-X_train_mean)/X_train_std

print("\nMean of normalized training data:", np.mean(X_train_norm, axis=0))
print("Std normalized data:", np.std(X_train_norm,axis=0))


# 5. Initialize the Neural Network Parameters
np.random.seed(42)

#w--> weight: The "strength" of the connections between neurons.The model adjusts these to learn
#b-->biases:The "threshold" the neuron has to cross to fire.
#layer 1: input (4)--> hidden (8)
# w1 shape: (4,8),b1 shape:(1,8)--->8 to 16(more neurons,more capicity to leran complex patterns)
#by He installation
w1 = np.random.randn(4,16) * 0.1
b1 = np.zeros((1,16))

#layer 2: hidden (8) --> output (3)-->8 to 16
#by He installation
w2 = np.random.randn(16,3) * 0.1
b2 = np.zeros((1,3))

print(f"w1 shape: {w1.shape}, b1 shape: {b1.shape}")
print(f"w2 shape: {w2.shape}, b2 shape: {b2.shape}")



# 6. Define Activation Functions
def relu(Z):
    return np.maximum(0.01 * Z, Z)
 #Leaky ReLU: lets a tiny bit of negative signal through
def softmax(Z): #soft max takes raw no. and squashes them into probabilities
    #subtract max for numerical stability to prevent overflow
    exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
    return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)

# 7. The Forward Pass
# Step A: Input-Hidden Layer
Z1 = np.dot(X_train_norm, w1) + b1  #broadcasting,np.dot for matrix multiplation
A1 = relu(Z1)

# Step B: Hidden Layer-Output Layer
Z2 = np.dot(A1, w2) + b2
A2 = softmax(Z2)

# A2 holds the probabilities for the 3 classes.
print(f"Shape of predictions (A2): {A2.shape}")
print("\nFirst 5 predictions (probabilities):")
print(A2[:5])

# 8. Loss Function and Backpropagation

# First, we need to one-hot encode our labels. 
# Right now y_train is just [0, 1, 2]. We need it to be [1, 0, 0], [0, 1, 0],... etc.
def one_hot(y, num_classes=3):
    one_hot_y = np.zeros((len(y), num_classes))
    one_hot_y[np.arange(len(y)), y] = 1
    return one_hot_y

y_train_onehot = one_hot(Y_train)

# Calculate Cross-Entropy Loss
# We add a tiny number (1e-8) to prevent log(0) which is undefined
m = len(Y_train) # Number of training samples (120)
loss = -np.sum(y_train_onehot * np.log(A2 + 1e-8)) / m
print(f"Initial Loss: {loss:.4f}")

# --- BACKPROPAGATION (The Calculus) ---
# Step 1: Output layer gradients
dZ2 = A2 - y_train_onehot
dw2 = np.dot(A1.T, dZ2) / m
db2 = np.sum(dZ2, axis=0, keepdims=True) / m


# Step 2: Hidden layer gradients
dA1 = np.dot(dZ2, w2.T) # The error derivative. It calculates the difference between the predicted probability and the actual label.
dZ1 = dA1 * (Z1 > 0)  # Derivative of ReLU (gradients)
dw1 = np.dot(X_train_norm.T, dZ1) / m  # Dot product with the Transpose. This aligns the dimensions so we can multiply matrices during backprop.
db1 = np.sum(dZ1, axis=0, keepdims=True) / m

# 9. Update Weights using Gradient Descent
learning_rate = 0.1
w1 -= learning_rate * dw1  #Gradient Descent. We subtract the gradient to go "downhill" and minimize the loss.
b1 -= learning_rate * db1
w2 -= learning_rate * dw2
b2 -= learning_rate * db2

# 10. THE TRAINING LOOP
learning_rate = 0.001 #decreased from 0.1 for effective learning
epochs = 4500 #increasing from 1000(initially 1000)

for epoch in range(epochs):
    # --- Forward Pass ---
    Z1 = np.dot(X_train_norm, w1) + b1
    A1 = relu(Z1)
    Z2 = np.dot(A1, w2) + b2
    A2 = softmax(Z2)
    
    # --- Calculate Loss ---
    loss = -np.sum(y_train_onehot * np.log(A2 + 1e-8)) / m
    
    # --- Backpropagation ---
    dZ2 = A2 - y_train_onehot
    dW2 = np.dot(A1.T, dZ2) / m
    db2 = np.sum(dZ2, axis=0, keepdims=True) / m
    
    dA1 = np.dot(dZ2, w2.T)
    dZ1 = dA1 * (Z1 > 0)
    dW1 = np.dot(X_train_norm.T, dZ1) / m
    db1 = np.sum(dZ1, axis=0, keepdims=True) / m
    
    # --- Update Weights ---
    w1 -= learning_rate * dw1
    b1 -= learning_rate * db1
    w2 -= learning_rate * dw2
    b2 -= learning_rate * db2
    
    # Print loss(ie error occured) every 500 epochs to watch it learn
    if epoch % 500 == 0:
        # Calculate training accuracy
        train_preds = np.argmax(A2, axis=1)
        train_acc = np.mean(train_preds == Y_train) * 100
        print(f"Epoch {epoch}, Loss: {loss:.4f}, Train Acc: {train_acc:.2f}%")

# 11. TESTING THE NETWORK
# Forward pass on the test data we hid earlier (the 30 unseen flowers)
Z1_test = np.dot(X_test_norm, w1) + b1
A1_test = relu(Z1_test)
Z2_test = np.dot(A1_test, w2) + b2
A2_test = softmax(Z2_test)

# Convert probabilities to actual predictions (0, 1, or 2)
# np.argmax finds the index of the highest probability
predictions = np.argmax(A2_test, axis=1)

# Calculate Accuracy
accuracy = np.mean(predictions == Y_test) * 100
print(f"\nFinal Test Accuracy: {accuracy:.2f}%")