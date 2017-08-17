#CNN

#Building the CNN

#Importing libraries
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator
#Initialize CNN
classifier = Sequential()

#Adding Convolutional layer
#Number of filters, then a tuple of number of rows and cols
classifier.add(Convolution2D(32, (3, 3) , input_shape = (64, 64, 3), activation = 'relu'))

#Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

#Adding a second convolutional layer
#No input shape required now
classifier.add(Convolution2D(32, (3, 3) , activation = 'relu'))

#Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

#Flattening
classifier.add(Flatten())

#Fully Connected Layer
#Hidden Layer
classifier.add(Dense(output_dim = 128, activation = 'relu'))
#Output Layer
classifier.add(Dense(output_dim = 1, activation = 'sigmoid'))

#Compiling the CNN
#Binary outcome that's why the loss is binary_crossentropy otherwise that changes
#categorical_crossentropy loss in case of 3 or more categories of outputs
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#Fitting CNN to images
#Used to augment images in the training set
train_datagen = ImageDataGenerator(
        rescale=1./255, #pixel values scaling from zero to one
        shear_range=0.2, 
        zoom_range=0.2,
        horizontal_flip=True) #images will be flipped horizontally

test_datagen = ImageDataGenerator(rescale=1./255)

#CNN trained on these images
training_set = train_datagen.flow_from_directory(
        'dataset/training_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

test_set = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

classifier.fit_generator(
        training_set,
        steps_per_epoch = 8000/32,
        nb_epoch=25,
        validation_data=test_set,
        validation_steps=2000/32)

# Making new predictions
import numpy as np
from keras.preprocessing import image
test_image = image.load_img('200_s.gif', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
training_set.class_indices
if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'
