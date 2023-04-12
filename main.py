import neural_network
import matplotlib.pyplot as plt
import time
import math

t = neural_network.extract_data("training_set.txt")
weights = [-0.4097906974405557, 2.074234291804565, -3.840499050301745, 3.1659349203041796]
grad = []
for i in range(len(weights)):
    grad.append(0)


steps = 1e1
learning_rate = 1e-5
t0 = time.time_ns()
loss_y = []
for rounds in range(int(steps)):
    loss = 0
    for i in range(len(t[0])):
        [temp_grad, temp_loss] = neural_network.training(t[0][i], weights, t[1][i])
        loss += temp_loss
        for i in range(len(grad)):
            grad[i] += temp_grad[i]

    if rounds % 100 == 0 and rounds != 0:
        print(rounds, round(loss, 4), grad)
        t1 = time.time_ns() - t0
        t1 *= 1e-9
        remaining = t1 / rounds * steps - t1
        r_text = str(int(remaining) // 3600)
        r_text += ":"
        r_text += str((int(remaining) // 60) % 60)
        r_text += ":"
        r_text += str(int(remaining) % 60)
        print("Time remaining: ", r_text)
        loss_y.append(loss)
    
    for i in range(len(grad)):
            weights[i] += grad[i] * learning_rate
            grad[i] = 0

print(weights)
plt.plot(loss_y)
plt.show()

t = neural_network.extract_data("test_set.txt")
data_in = t[0][0]

y = []

for i in range(1000):
    y_pred = neural_network.calculate_output(data_in, weights)
    data_in.pop(0)
    data_in.append(y_pred)
    y.append(y_pred)

plt.plot(y)
plt.show()

weights = [-0.4097906974405557, 2.074234291804565, -3.840499050301745, 3.1659349203041796]
err = 0
for i in range(len(t[0])):
    err += math.fabs(neural_network.calculate_output(t[0][i], weights) - t[1][i]) / len(t[0])

print(err)
     