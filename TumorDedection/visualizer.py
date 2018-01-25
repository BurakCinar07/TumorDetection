import matplotlib.pyplot as plt
import json

data = json.load(open("data.json"))
x_axis = list(range(len(data['batch_accuracies'])))
y_axis = data['batch_accuracies']
print(data['batch_accuracies'])
plt.xlabel("batch")
plt.ylabel("accuracy")
plt.plot(x_axis, y_axis)
plt.show()