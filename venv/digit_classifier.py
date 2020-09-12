from models import sudoku_cnn
import argparse
from sklearn.metrics import classification_report
from sklearn.metrics import classification_report
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import mnist
from sklearn.preprocessing import LabelBinarizer

arg = argparse.ArgumentParser()
arg.add_argument("-m", "--model", required=True, help="Path to output model after training")
args = vars(arg.parse_args())

LR = 1e-3
EPOCHS = 10
BATCH_SIZE = 128

print("Accessing MNIST")
((X_train, y_train), (X_test, y_test)) = mnist.load_data()

X_train = X_train.reshape((X_train.shape[0], 28, 28, 1))
X_test = X_test.reshape((X_test.shape[0], 28, 28, 1))

X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0

label = LabelBinarizer()
y_train = label.fit_transform(y_train)
y_test = label.transform(y_test)

opt = Adam(lr=LR)
model = sudoku_cnn.Sudoku.build(w=28, h=28, depth=1, classes=10)
model.compile(loss="categorical_crossentropy", optimizer=opt,metrics=["accuracy"])

H = model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=BATCH_SIZE, epochs=EPOCHS,verbose=1)

predictions = model.predict(X_test)
print(classification_report(y_test.argmax(axis=1), predictions.argmax(axis=1), target_names=[str(x) for x in label.classes_]))

model.save(args["model"], save_format="h5")










