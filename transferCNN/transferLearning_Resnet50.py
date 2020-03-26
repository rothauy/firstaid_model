#keras setup
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from keras.optimizers import SGD

def createResNet50Model():
    # create model for ResNet50
    base_model = ResNet50(weights = 'imagenet', include_top=False)

    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer
    x = Dense(1024, activation='relu')(x)
    # and a logistic layer -- let's say we have 2 classes
    predictions = Dense(2, activation='softmax')(x)

    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)

    # first: train only the top layers (which were randomly initialized)
    for layer in base_model.layers:
        layer.trainable = False

    # compile the model (should be done *after* setting layers to non-trainable)
    model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy',metrics=['accuracy'])

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
    # weight_path = ('')
    model = createResNet50Model()

    # Training model for 5 epochs
    model.fit_generator(
        train_generator,
        steps_per_epoch= 531 // batch_size,
        epochs=5,
        validation_data=validation_generator,
        validation_steps= 183 // batch_size)

    # Unfreeze the last stage of ResNet50 to transfer learning
    for layer in model.layers[0:143]:
        layer.trainable = False
    
    for layer in model.layers[143:]:
        layer.trainable = True

    #run
    model.fit_generator(
        train_generator,
        steps_per_epoch= 531// batch_size,
        epochs=50,
        validation_data=validation_generator,
        validation_steps= 183// batch_size)

    #evaluating the model
    score = model.evaluate_generator(test_generator, 177//batch_size)
    print("Loss: ", score[0], "Accuracy: ", score[1])

    # always save your weights after training or during training
    model.save_weights('findWound_ResNet50_Transfer_Learning_weight.h5')

    # save the model
    model.save('findWound_ResNet50_Transfer_Learningt_model.h5')
    

main()
