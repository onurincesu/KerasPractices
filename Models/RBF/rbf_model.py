import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import StandardScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Layer, Dense, Flatten
from tensorflow.keras import backend as K

import warnings
warnings.filterwarnings("ignore")

iris = load_iris()
X = iris.data
y = iris.target

label_binarizer = LabelBinarizer()
y_encoded = label_binarizer.fit_transform(y)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size = 0.2, random_state=42 )

class RBFLayer(Layer):
    def __init__(self, units, gamma, **kwargs):
        super(RBFLayer, self).__init__(**kwargs)
        self.units = units
        self.gamma = K.cast_to_floatx(gamma)
    def build(self, input_shape):
        self.mu = self.add_weight(name = "mu",
                                    shape = (int(input_shape[1]), self.units),
                                    initializer = "uniform",
                                    trainable = True
                                    )
        super(RBFLayer, self).build(input_shape)
    def call(self, inputs):
        diff = K.expand_dims(inputs) - self.mu
        l2 = K.sum(K.pow(diff, 2), axis = 1)
        res = K.exp(-1 * self.gamma * l2)
        return res
    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.units)


def build_model():
    model= Sequential()
    model.add(Flatten(input_shape=(4,)))
    model.add(RBFLayer(10, 0.5))
    model.add(Dense(3, activation = "softmax"))
    model.compile(optimizer = "adam",
                    loss = "categorical_crossentropy",
                    metrics = ["accuracy"])
    return model

model = build_model()

history = model.fit(X_train, y_train,
                    epochs = 2000,
                    batch_size = 4,
                    validation_split = 0.3,
                    verbose = 1)


loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")

plt.figure()
plt.subplot(1, 2, 1)
plt.plot(history.history["loss"], marker = "o",label = "Training Loss")
plt.plot(history.history["val_loss"], marker = "o", label = "Validation Loss")
plt.title("Training and Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.grid(True)
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history["accuracy"], marker = "o", label = "Training Accuracy")
plt.plot(history.history["val_accuracy"], marker = "o", label = "Validation Accuracy")
plt.title("Training and Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.grid(True)
plt.legend()
plt.show()
