import numpy as np
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_test = x_test.astype(np.float32) / 255.0

# Ghi 10 ảnh đầu tiên
np.savetxt("X.txt", x_test[:10].reshape(-1), fmt="%.6f") #nếu muốn lấy n ảnh đầu tiên thì chỉnh  x_test[:n]
np.savetxt("Y.txt", y_test[:10], fmt="%d") #nếu muốn lấy n label đầu tiên thì chỉnh y_test[:n] 