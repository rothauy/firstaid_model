# # keras setup
from keras.models import Model, Sequential
from keras.layers import Dense, GlobalAveragePooling2D, Activation, Conv2D, MaxPooling2D, Lambda, Flatten, Add
from keras.layers.merge import concatenate
from keras.preprocessing.image import ImageDataGenerator
from keras.initializers import RandomNormal
from keras import backend as K
from keras.optimizers import SGD, Adam, RMSprop
from keras.regularizers import l2
from keras.losses import binary_crossentropy
from keras import applications, Input

# other modules
import os
import numpy as np
from PIL import Image
from cv2 import imread, resize
import pickle
import sys
import numpy.random as rand

# limiting GPUs access
# import os
# os.environ["CUDA_VISIBLE_DEVICES"]="1,2"

# creating the model
def buildModel(input_shape):

    # Define the tensors for two input
    left = Input(input_shape)
    right = Input(input_shape)

    model = Sequential()

    kernel_initializer = RandomNormal(0, 1e-2)
    bias_initializer = RandomNormal(0.5, 1e-2)
    kernel_regularizer_l2 = l2(2e-4)

    model.add(Conv2D(64, (10, 10), activation='relu', kernel_initializer=kernel_initializer, bias_initializer= bias_initializer, kernel_regularizer=kernel_regularizer_l2, input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (7, 7), activation='relu', kernel_initializer=kernel_initializer, bias_initializer= bias_initializer, kernel_regularizer=kernel_regularizer_l2, input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(256, (4, 4), activation='relu', kernel_initializer=kernel_initializer, bias_initializer= bias_initializer, kernel_regularizer=kernel_regularizer_l2, input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
    model.add(Dense(4096, activation='sigmoid', kernel_initializer=kernel_initializer, bias_initializer= bias_initializer, kernel_regularizer=l2(2e-3) ))

    # generete the encoding, feature vectors, for two images
    encoded_l = model(left)
    encoded_r = model(right)

    # merge two encoded inputs with the l1 distance
    L1_distance = lambda x: K.abs(x[0]-x[1])
    merged = Lambda(L1_distance)([encoded_l, encoded_r])

    prediction = Dense(1, activation='sigmoid', bias_initializer=bias_initializer)(merged)

    # define Siamese Network model
    siamese_model = Model(input=[left,right], output=prediction)

    # setup Adam optimizer - learning rate (default 10^-3) -> medain freq weighting should be .0006
    optimizer = Adam(lr = 0.00006)

    # the model so far outputs 3D feature maps (height, width, features)
    siamese_model.compile(loss='binary_crossentropy', optimizer= optimizer, metrics=['accuracy'])

    siamese_model.summary()

    return siamese_model

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

# function to prepare date for Siamese Network
def loadImages(path, n = 0):
    x = []
    y = []
    woundDict = {}
    curr = n

    # loading different type of wound for the isolation
    for woundType in os.listdir(path):
        woundDict[woundType] = [curr, None]
        woundTypePath = os.path.join(path,woundType)
        # loading each wound and insert them into their column in the array
        for eachType in os.listdir(woundTypePath):
            categoryImages = []
            eachPath = os.path.join(woundTypePath, eachType)
            for each in os.listdir(eachPath):
                imagePath = os.path.join(eachPath,each)
                origImage = imread(imagePath)
                image = resize(origImage,(250,250))
                categoryImages.append(image)
                y.append(curr)
            try:
                x.append(np.stack(categoryImages))
                print (x[curr].shape)
            except ValueError as e:
                print (e)
                print ("error - catagory images: ", categoryImages)
            woundDict[woundType][1] = curr
            curr = curr + 1

    y = np.vstack(y)
    x = np.stack(x)
    print (x.shape)
    return x, y, woundDict

# function to create the pair for model
def get_batch(batch_size, X):
    n_classes, n_samples, w, h, d = X.shape
    categories = rand.choice(n_classes,size=(batch_size,),replace=False)
    pairs=[np.zeros((batch_size, h, w, 3)) for i in range(2)]

    targets=np.zeros((batch_size,))
    targets[batch_size//2:] = 1
    for i in range(batch_size):
        category = categories[i]
        idx_1 = rand.randint(0, n_samples)
        pairs[0][i,:,:,:] = X[category, idx_1].reshape(w, h, d)
        idx_2 = rand.randint(0, n_samples)

        if i >= batch_size // 2:
            category_2 = category  
        else: 
            category_2 = (category + rand.randint(1,n_classes)) % n_classes
        pairs[1][i,:,:,:] = X[category_2,idx_2].reshape(w, h,d)
    return pairs, targets
    
def generate(batch_size, X):
    while True:
        pairs, targets = get_batch(batch_size,X)
        yield (pairs, targets)    

# function to run CNN
def main():

    # variable
    batch_size = 4
    input_3d = (250,250,3)
    trPath = 'C:\\Users\\Rotha Uy\\Documents\\firstaid_model\\firstaid_model\\Dataset\\training'
    vPath = 'C:\\Users\\Rotha Uy\\Documents\\firstaid_model\\firstaid_model\\Dataset\\validating'
    tstPath = 'C:\\Users\\Rotha Uy\\Documents\\firstaid_model\\firstaid_model\\Dataset\\testing'

    # sys.stdout = open('out.dat', 'w')
    # dataset preparation
    # train_generator, validation_generator, test_generator = generateData(batch_size,input_shape)
  
    # preparing data for training and validating
    (trData, trY, trClasses) = loadImages(trPath)
    (vData, vY, vClasses) = loadImages(vPath)
    (tstData, tstY, tstClasses) = loadImages(tstPath)

    # fix the weight and build the model
    # weight_path = ('')
    model = buildModel(input_3d)

    #run
    model.fit_generator(
        generate(batch_size,trData),
        steps_per_epoch= 207// batch_size,
        epochs=50,
        validation_data=generate(batch_size,vData),
        validation_steps= 62// batch_size)

    # # #evaluating the model
    # # score = model.evaluate_generator(test_generator, 62//batch_size)
    # # print("Loss: ", score[0], "Accuracy: ", score[1])

    # # always save your weights after training or during training
    # model.save_weights('findWound_OneShot_Siamese_weight.h5')

    # # save the model
    # model.save('findWound_OneShot_Siamese_model.h5')
    # sys.stdout.close()

main()