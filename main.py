from Classes import *
import pickle


with open("WineClassifierTron3000.pkl", "rb") as f:
    model = pickle.load(f)

Net = Network()
Net.load_state(model["state"])

scaler=model["scaler"]


#predict
X = [
13.32,
3.24,
2.38,
21.5,
92.0,
1.93,
0.76,
0.45,
1.25,
8.42,
0.55,
1.62,
650.0
]

Xin=scaler.transform([X])[0]
out=argmax(softmax(Net.forward(Xin)))

print(out)