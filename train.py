from Classes import *
import matplotlib.pyplot as plt
import pickle
import time
from sklearn.datasets import load_wine


wine = load_wine() # 20 cases of output 0,1 and 2 each
X = (
    list(wine.data[0:40]) +
    list(wine.data[59:99]) +
    list(wine.data[130:170])
)

Y = (
    list(wine.target[0:40]) +
    list(wine.target[59:99]) +
    list(wine.target[130:170])
)


scaler = StdScaler()
X = scaler.fit_transform(X)

Net=Network()

print('Starting')
t=time.time()
Xloss,Yloss=Train(Net,X,Y)
print(f'Finished. Time taken: {time.time()-t}')

# Accuracy check
correct = 0

for x, y in zip(X, Y):
    pred = argmax(softmax(Net.forward(x)))
    if pred == y:
        correct += 1

print(f"Accuracy: {correct}/{len(X)}")


plt.title('Loss per 10 Iterations')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.plot(Xloss,Yloss)
plt.show()

model={"state":Net.get_state() ,"scaler":scaler}

with open("WineClassifierTron3000.pkl", "wb") as f: #storing model
    pickle.dump(model,f)
print("Model saved successfully")


