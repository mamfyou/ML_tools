import numpy as np
import tensorflow as tf
from keras import Sequential
from keras.src.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from keras.src.utils import load_img, img_to_array

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

train_set = train_datagen.flow_from_directory(
    '../../data_sets/images/training_set',
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary'
)

test_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test_set = test_datagen.flow_from_directory(
    '../../data_sets/images/test_set',
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary'
)

cnn = Sequential()
# adding first convolutional layer
cnn.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(64, 64, 3)))
# adding max pooling layer
cnn.add(MaxPooling2D(pool_size=2, strides=2))
# adding second convolutional layer
cnn.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
cnn.add(MaxPooling2D(pool_size=2, strides=2))
# flattening
cnn.add(Flatten())
# full connection
cnn.add(Dense(128, activation='relu'))
# output layer
cnn.add(Dense(1, activation='sigmoid'))

cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

cnn.fit(x=train_set, validation_data=test_set, epochs=25)
# cnn.save('cnn.h5')
# cnn.evaluate(test_set)
# cnn.save('cnn.h5')
# cnn.save_weights('cnn.h5')

# making single prediction
test_image = load_img('../../data_sets/images/test_set/dogs/dog.4001.jpg', target_size=(64, 64))
test_image = img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
result = cnn.predict(test_image)

if result[0][0] == 1:
    print('Dog')
else:
    print('Cat')
