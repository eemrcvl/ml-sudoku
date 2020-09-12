import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, BatchNormalization, MaxPool2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

#import dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

print(X_train.shape) #check if correctly imported

X_train = X_train.reshape((-1, 28, 28, 1))
X_test = X_test.reshape((-1, 28, 28, 1))

X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0

y_train = to_categorical(y_train, num_classes=10)

datagen = ImageDataGenerator(rotation_range=10, zoom_range=0.1, height_shift_range=0.1, width_shift_range=0.1)

#view augmented images
X_train2 = X_train[9,].reshape((1,28,28,1))
y_train2 = y_train[9,].reshape((1, 10))
plt.figure(figsize=(10, 5))
for i in range(30):
    plt.subplot(3, 10, i + 1)
    X_train3, y_train3 = datagen.flow(X_train2, y_train2).next()
    plt.imshow(X_train3[0].reshape((28, 28)), cmap=plt.cm.binary)
    plt.axis('off')
    if i == 9: X_train2 = X_train[11,].reshape((1, 28, 28, 1))
    if i == 19: X_train2 = X_train[18,].reshape((1, 28, 28, 1))

plt.subplots_adjust(wspace=-0.1, hspace=-0.1)
plt.show()

#Convolutional Neural Network - LeNet5 architecture


model = Sequential()
model.add(Conv2D(32, kernel_size=3, activation='relu', input_shape=(28, 28, 1)))
model.add(BatchNormalization())
model.add(Conv2D(32, kernel_size=3, activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(32, kernel_size=5, strides=2, padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))

model.add(Conv2D(64, kernel_size=3, activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(64, kernel_size=3, activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(64, kernel_size=5, strides=2, padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))

model.add(Conv2D(128, kernel_size=4, activation='relu'))
model.add(BatchNormalization())
model.add(Flatten())
model.add(Dropout(0.4))
model.add(Dense(10, activation='softmax'))

#Compile
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])


lr = LearningRateScheduler(lambda x: 1e-3 * 0.95 ** x)
history = [0]
epochs = 45
X_train2, X_val2, y_train2, y_val2 = train_test_split(X_train, y_train, test_size=0.1)
history = model.fit_generator(datagen.flow(X_train2, y_train2, batch_size=64), epochs=epochs, steps_per_epoch=X_train2.shape[0] // 64, validation_data=(X_val2, y_val2), callbacks=[lr], verbose=0)
model.save("MNIST_classifier_v2", save_format="h5")

result =   model.predict(X_test)

result = np.argmax(result, axis = 1)
result = pd.Series(result, name="Label")
submission = pd.concat([pd.Series(range(1, 28001), name="ImageId"), result], axis=1)
submission.to_csv("MNIST_CNN.csv", index=False)
