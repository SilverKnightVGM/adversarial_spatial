import math, json, os, sys
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import keras
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from tensorflow.keras.applications.resnet50 import preprocess_input

# from keras.layers import Dense, Flatten, GlobalAveragePooling2D
from keras.layers.core import Dense, Flatten
from keras.layers.pooling import GlobalAveragePooling2D

from keras.models import Model
from keras.optimizers import Adam, SGD
from keras.preprocessing import image
from keras.regularizers import l2
import platform
import multiprocessing
# from keras import backend as K
# if K.backend()=='tensorflow':
    # K.set_image_dim_ordering("th")
    
def add_to_dict(dictionary, *argv):
    for a in argv:
        my_var_name = [ k for k,v in globals().items() if v == a][0]
        dictionary[str(my_var_name)] = a

# Don't print warnings
import os, logging

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)

if (platform.system() == "Windows"):
    patience_num_epochs = 5
else:
    patience_num_epochs = 100

num_workers = int(multiprocessing.cpu_count() * 6/8)
SIZE = (32, 32)
INPUT_SHAPE = (32,32,3)
BATCH_SIZE = 32
EPOCHS = 600
LEARNING_RATE = 0.0001
# DECAY = 0.0002
MOMENTUM = 0.9

all_params = {}
# add_to_dict(all_params, BATCH_SIZE, EPOCHS, LEARNING_RATE, DECAY, MOMENTUM)
add_to_dict(all_params, BATCH_SIZE, EPOCHS, LEARNING_RATE, MOMENTUM)
print(all_params)

base_dir = os.path.dirname(os.path.abspath(__file__))

# Images inside subfolders for each class
DATA_DIR = 'images-keras-original'
DATA_DIR = os.path.join("..", DATA_DIR)

TRAIN_DIR = os.path.abspath(os.path.join(base_dir, DATA_DIR, 'train'))
VALID_DIR = os.path.abspath(os.path.join(base_dir, DATA_DIR, 'test'))

if __name__ == "__main__":
    num_train_samples = sum([len(files) for r, d, files in os.walk(TRAIN_DIR)])
    num_valid_samples = sum([len(files) for r, d, files in os.walk(VALID_DIR)])

    # num_train_steps = math.floor(num_train_samples/BATCH_SIZE) // 5
    num_train_steps = math.floor(num_train_samples/BATCH_SIZE)
    num_valid_steps = math.floor(num_valid_samples/BATCH_SIZE)
    print("num_train_steps: {}, num_valid_steps:{}".format(num_train_steps, num_valid_steps))
    # num_train_steps = 100
    # num_valid_steps = 500

    # gen = keras.preprocessing.image.ImageDataGenerator(horizontal_flip=True)
    # val_gen = keras.preprocessing.image.ImageDataGenerator()
    
    # gen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255, horizontal_flip=True)
    # val_gen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    
    gen = keras.preprocessing.image.ImageDataGenerator(preprocessing_function=preprocess_input, horizontal_flip=True)
    val_gen = keras.preprocessing.image.ImageDataGenerator(preprocessing_function=preprocess_input)
    
    # Do this also in inference to predict properly
    # gen = keras.preprocessing.image.ImageDataGenerator(rescale=1./138, horizontal_flip=True)
    # val_gen = keras.preprocessing.image.ImageDataGenerator(rescale=1./138)

    batches = gen.flow_from_directory(TRAIN_DIR, color_mode='rgb', target_size=SIZE, class_mode='categorical', shuffle=True, batch_size=BATCH_SIZE)
    val_batches = val_gen.flow_from_directory(VALID_DIR, color_mode='rgb', target_size=SIZE, class_mode='categorical', shuffle=True, batch_size=BATCH_SIZE)
    
    classes = list(iter(batches.class_indices))
    print(classes)

    finetuned_model = keras.applications.resnet50.ResNet50(input_shape=INPUT_SHAPE, include_top=True, weights=None, classes=len(classes))

    finetuned_model.summary()

    # finetuned_model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    # finetuned_model.compile(optimizer=SGD(lr=LEARNING_RATE, decay=DECAY, momentum=MOMENTUM, nesterov=True, clipvalue=0.5), loss='categorical_crossentropy', metrics=['accuracy'])
    finetuned_model.compile(optimizer=SGD(lr=LEARNING_RATE, momentum=MOMENTUM, nesterov=True), loss='categorical_crossentropy', metrics=['accuracy'])
    for c in batches.class_indices:
        classes[batches.class_indices[c]] = c
    finetuned_model.classes = classes

    early_stopping = EarlyStopping(patience=patience_num_epochs)
    tensorboard = TensorBoard(log_dir=os.path.join(base_dir, 'logs'))
    
    # model_name = 'resnet50_top_best.h5'
    # model_name = 'resnet50_top_{epoch:03d}'
    model_name = 'resnet50_top_best'
    filepath = os.path.join(base_dir, model_name + ".h5")
    checkpointer = ModelCheckpoint(filepath=filepath, verbose=1, save_best_only=True)

    history = finetuned_model.fit_generator(batches, 
                                            steps_per_epoch=num_train_steps, 
                                            epochs=EPOCHS, 
                                            callbacks=[early_stopping, checkpointer, tensorboard], 
                                            validation_data=val_batches, 
                                            validation_steps=num_valid_steps,
                                            workers=num_workers)
    # finetuned_model.save('resnet50_final.h5')
    score = finetuned_model.evaluate_generator(val_batches, steps=num_valid_steps)

    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    # print(history.history.keys())

    print('Training loss:', history.history['loss'][-1])
    print('Training accuracy:', history.history['acc'][-1])

    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='best')
    plt.title('Model Accuracy ' + model_name)
    # plt.savefig('model_acc.png')
    plt.savefig(os.path.join(base_dir,'model_acc_{}.png'.format(model_name)))
    plt.figure()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='best')
    plt.title('Model Loss ' + model_name)
    # plt.savefig('model_loss.png')
    plt.savefig(os.path.join(base_dir,'model_loss_{}.png'.format(model_name)))