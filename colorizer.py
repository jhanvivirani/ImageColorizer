import tensorflow as tf
from tensorflow.keras import layers

model = tf.keras.Sequential([
    layers.Flatten(9),
    layers.Dense(64, activation=tf.nn.relu),
    layers.Dense(64, activation=tf.nn.relu),
    layers.Dense(3, activation=tf.nn.sigmoid)
])

model.compile(optimizer=tf.train.AdamOptimizer(), 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(grayScale, middlePixel, epochs=5) #grayscale = training b&w window , middelPixel = rgb value of middle pixel

accuracy = model.evaluate(testingGray , testingMiddlePixel)
print(accuracy)

#use pixel = model.predict(window) and multiply each value by 255 to find pixel rgb value