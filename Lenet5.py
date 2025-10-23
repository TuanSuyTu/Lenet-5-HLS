#5k8 98,6%
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# --- 1. TẢI VÀ CHUẨN BỊ DỮ LIỆU ---

# Tải bộ dữ liệu MNIST
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Chuẩn hóa dữ liệu ảnh: đưa giá trị pixel từ [0, 255] về [0, 1]
# Đây là bước rất quan trọng để model hội tụ tốt hơn
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Thêm một chiều "channel" vào ảnh (MNIST là ảnh xám nên chỉ có 1 channel)
# Kích thước từ (60000, 28, 28) -> (60000, 28, 28, 1)
x_train = tf.expand_dims(x_train, -1)
x_test = tf.expand_dims(x_test, -1)

# Chuyển nhãn (label) sang dạng one-hot encoding
# Ví dụ: số 3 -> [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
num_classes = 10
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# --- 2. XÂY DỰNG MODEL ---

# Sử dụng Keras Sequential API để xây dựng model từng lớp một
model = keras.Sequential([
    keras.Input(shape=(28, 28, 1)),

    # Lớp tích chập 1
    layers.Conv2D(4, kernel_size=(5, 5), activation="relu"),
    layers.MaxPooling2D(pool_size=(2, 2)),

    # Lớp tích chập 2
    layers.Conv2D(8, kernel_size=(5, 5), activation="relu"),
    layers.MaxPooling2D(pool_size=(2, 2)),

    # Chuyển 2D -> 1D
    layers.Flatten(),

    # Dropout để tránh overfitting
    layers.Dropout(0.2),

    # Dense layer 1
    layers.Dense(32, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(16, activation="relu"),
    

    # Dense layer 2 (đầu ra)
    layers.Dense(num_classes, activation="softmax"),
])

# In ra cấu trúc và số lượng tham số của model
# Đây là một công cụ cực kỳ hữu ích của Keras!
print("Cấu trúc model:")
model.summary()

# --- 3. BIÊN DỊCH (COMPILE) MODEL ---

# Cấu hình optimizer, loss function và metrics cho model
model.compile(
    optimizer="adam", # Optimizer phổ biến và hiệu quả
    loss="categorical_crossentropy", # Loss function phù hợp cho bài toán phân loại one-hot
    metrics=["accuracy"] # Theo dõi độ chính xác
)

# --- 4. HUẤN LUYỆN (FIT) MODEL ---

batch_size = 64
epochs = 40 

print("\nBắt đầu huấn luyện model...")
# Quá trình huấn luyện sẽ diễn ra ở đây
history = model.fit(
    x_train, y_train,
    batch_size=batch_size,
    epochs=epochs,
    validation_split=0.1 # Dùng tập test để đánh giá sau mỗi epoch
)

# --- 5. ĐÁNH GIÁ MODEL ---

print("\nĐánh giá model trên tập dữ liệu test...")
score = model.evaluate(x_test, y_test, batch_size=batch_size)
print(f"Test loss: {score[0]:.4f}")
print(f"Test accuracy: {score[1]:.4f} ({score[1]*100:.2f}%)")

model.save("model.h5")