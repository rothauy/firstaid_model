#keras setup
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import backend as K
from tensorflow.keras import applications
from tensorflow.keras.initializers import VarianceScaling
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
#other modules
import os, glob, random
import importlib
import numpy as np
from PIL import Image

# # limiting GPUs access
# import os
# os.environ["CUDA_VISIBLE_DEVICES"]="1,2"

# creating the model -- Simple CNN
def buildModel():

    model = Sequential()

    kernel_initializer = keras.initializers.VarianceScaling(scale=1.0, mode='fan_in', distribution='normal', seed=None)
    model.add(Conv2D(8, (3, 3), kernel_initializer=kernel_initializer, input_shape=(150, 150, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(16, (3, 3), kernel_initializer=kernel_initializer))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3), kernel_initializer=kernel_initializer))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3), kernel_initializer=kernel_initializer))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3), kernel_initializer=kernel_initializer))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
    model.add(Dense(64, kernel_initializer=kernel_initializer))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(3))
    model.add(Activation('softmax'))

    optimizer = Adam(lr=0.00006)

    # # load pre-train weights
    # model.load_weights(weights_path)

    # set the first # layers (up to the last conv block)
    # to non-trainable (weights will not be updated)
    # for layer in model.layers[:1]:
    #     layer.trainable = False

    # the model so far outputs 3D feature maps (height, width, features)
    model.compile(loss='categorical_crossentropy',
                  optimizer= optimizer,
                  #learning rate (default 10^-3) -> medain freq weighting should be .0006
                  metrics=['accuracy'])

    return model

def generateData(batch_size,input_shape):
    # this is the augmentation configuration we will use for training
    train_datagen = ImageDataGenerator(
            rescale=1./255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True, 
            vertical_flip= True)

    # this is the augmentation configuration we will use for testing:
    # only rescaling
    test_datagen = ImageDataGenerator(rescale=1./255)

    # this is a generator that will read pictures found in
    # subfolers of 'data/train', and indefinitely generate
    # batches of augmented image data
    train_generator = train_datagen.flow_from_directory(
            'C:\\Users\\Rotha Uy\\Documents\\firstaid_model\\firstaid_model\\Dataset\\training',  # this is the target directory
            target_size=input_shape,  # all images will be resized to 150x150
            batch_size=batch_size,
            class_mode='categorical')  # since we use binary_crossentropy loss, we need binary labels

    # this is a similar generator, for validation data
    validation_generator = test_datagen.flow_from_directory(
            'C:\\Users\\Rotha Uy\\Documents\\firstaid_model\\firstaid_model\\Dataset\\validating',
            target_size=input_shape,
            batch_size=batch_size,
            class_mode='categorical')

    # this is a similar generator, for test data
    test_generator = test_datagen.flow_from_directory(
            'C:\\Users\\Rotha Uy\\Documents\\firstaid_model\\firstaid_model\\Dataset\\testing',
            target_size=input_shape,
            batch_size=batch_size,
            class_mode='categorical',
            shuffle = False,
            save_to_dir = 'C:\\Users\\Rotha Uy\\Documents\\firstaid_model\\firstaid_model\\Dataset\\generated\\testing',
            save_format = 'png')

    return train_generator, validation_generator, test_generator

# function to run CNN
def main():

    # variable
    batch_size = 16
    input_shape = (150,150)

    train_generator, validation_generator, test_generator = generateData(batch_size,input_shape)

    # fix the weight and build the model
    weight_path = ('')
    model = buildModel()

    #run
    model.fit_generator(
        train_generator,
        steps_per_epoch= 207// batch_size,
        epochs=50,
        validation_data=validation_generator,
        validation_steps= 62// batch_size)

    #evaluating the model
    score = model.evaluate_generator(test_generator, 62//batch_size)
    print("Loss: ", score[0], "Accuracy: ", score[1])

    # always save your weights after training or during training
    model.save_weights('findWound_simpleCNN_cut_weight.h5')

    # save the model
    model.save('findWound_simpleCNN_cut_model.h5')
    

main()