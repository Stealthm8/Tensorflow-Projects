import tensorflow as tf
import numpy as np
import random
import matplotlib.pyplot as plt

#load data into training and testing data
mnistData = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnistData.load_data()

class_names = ['0', '1', '2', '3', '4', '5', '6', '6', '7', '8', '9']
train_images = train_images / 255.0
test_images = test_images / 255.0

model = tf.keras.Sequential([tf.keras.layers.Flatten(input_shape = (28, 28)), tf.keras.layers.Dense(128, activation = 'relu'), tf.keras.layers.Dense(10)])
model.compile(optimizer = 'adam', loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True), metrics = ['accuracy'])
model.fit(train_images, train_labels, epochs=10)

test_loss, test_acc = model.evaluate(test_images,  test_labels)
probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
predictions = probability_model.predict(test_images)

for i in range(0,500):
	validation = " incorrect"
	if (np.argmax(predictions[i]) == test_labels[i]):
		validation = " correct"
	print("predicted label: " + str(np.argmax(predictions[i])) + " vs actual label: " + str(test_labels[i]) + validation)

print("\ntesting data provided loss: " + str(test_loss) + " and accuracy: " + str(test_acc))
