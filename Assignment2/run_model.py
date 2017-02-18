from spiral_model import spiral_model
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def generate_spiral(t, constant):
    spiral_x = constant * t * np.cos(t) + np.random.normal(scale=constant*0.1)
    spiral_y = constant * t * np.sin(t) + np.random.normal(scale=constant*0.1)
    return spiral_x, spiral_y

num_elements = 100
spiral_constant_1 = 0.2
spiral_constant_2 = 0.3
parameter = np.linspace(0, 3*np.pi, num_elements)
x_1, y_1 = generate_spiral(parameter, spiral_constant_1)
x_2, y_2 = generate_spiral(parameter+np.pi*0.5, spiral_constant_2)
c_1 = np.ones(num_elements)
c_2 = np.zeros(num_elements)
x = np.append(x_1, x_2)
y = np.append(y_1, y_2)
c = np.append(c_1, c_2)

layers = [10, 30, 50, 30, 20, 10]
rate = 0.003
iterations = 200
gamma = 2.5
model = spiral_model(tf.Session(), layers, rate, iterations, gamma)
model.initialize()
model.train(x, y, c)
print("Predicting class of training data:")
for x_p, y_p in zip(x_1, y_1):
    print(model.predict(x_p, y_p))

#Attempt at plotting....
#Thanks to Krishna for pointing me towards contours
#I just have no idea how to make that work.....
#But more likely than that it's because my model is ass
#domain = np.arange(-4, 4, 0.1)
#x_cont, y_cont = np.meshgrid(domain, domain)
#prob = np.vectorize(model.predict)
#z_cont = prob(x_cont, y_cont)
plt.plot(x_1, y_1, 'r^', x_2, y_2, 'b^')
plt.xlabel('x')
plt.ylabel('y')
plt.title("Spiral predictions with boundaries")
#plt.contour(x_cont, y_cont, z_cont, levels=[0.5])
PdfPages('what_am_i_doing.pdf').savefig()
plt.show()
