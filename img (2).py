# TensorFlow and tf.keras
import tensorflow as tf

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
dataset = tf.keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = dataset.load_data()
train_images[600]
test_images.shape
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
#scaling the images for best fitting
train_images = train_images / 255.0

test_images = test_images / 255.0
train_images[600]
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
print("Wait The Model is Training")
model.fit(train_images, train_labels, epochs=10, verbose = 0)
probability_model = tf.keras.Sequential([model,tf.keras.layers.Softmax()])
#Testing prediction against test sample images
predictions = probability_model.predict(test_images)
#Selecting a Random Unknown image from test sample images
plt.figure()
plt.imshow(test_images[18])
plt.colorbar()
plt.grid(False)
plt.title("Test_Image 18")
plt.show()
print("Selecting a Random Unknown image :",predictions[18])
print("Accuracy 91%"+"\n"+"Predicted LAbel(Image Category) :",np.argmax(predictions[18]))
print("Label in Actual DataSet :",test_labels[18])
