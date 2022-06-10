import tensorflow as tf
import matplotlib.pyplot as plt

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(len(x_train), len(y_train))
print(len(x_test), len(y_test))

x_train, x_test = x_train / 255.0, x_test / 255.0

plt.imshow(x_train[0])
plt.show()

