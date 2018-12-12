import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras import layers

grayScale = [[1,2,3,4,5,6,7,8,9],[12,24,34,4,52,63,74,85,978]]
middlePixel = [[1,2,3], [23,3,12]]
testingGray = grayScale
testingMiddlePixel = middlePixel

print("making model...")
model = Sequential([
    layers.Dense(10, input_dim = 9, activation=tf.nn.relu),
    layers.Dense(3, activation=tf.nn.sigmoid)
])

print("model made")
print("compiling model...")
model.compile(optimizer=tf.train.AdamOptimizer(), 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
print("Fitting")
model.fit(grayScale, middlePixel, epochs=5) #grayscale = training b&w window , middelPixel = rgb value of middle pixel
print("evaluating...")
accuracy = model.evaluate(testingGray , testingMiddlePixel)
print(accuracy)

#use pixel = model.predict(window) and multiply each value by 255 to find pixel rgb value