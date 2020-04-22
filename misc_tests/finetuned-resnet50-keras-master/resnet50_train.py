import math, json, os, sys
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import keras
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense, Flatten
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing import image


DATA_DIR = '../images-keras-original'
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
VALID_DIR = os.path.join(DATA_DIR, 'valid')
SIZE = (32, 32)
INPUT_SHAPE = (32,32,3)
BATCH_SIZE = 128
EPOCHS = 600


if __name__ == "__main__":
    num_train_samples = sum([len(files) for r, d, files in os.walk(TRAIN_DIR)])
    num_valid_samples = sum([len(files) for r, d, files in os.walk(VALID_DIR)])

    num_train_steps = math.floor(num_train_samples/BATCH_SIZE)
    num_valid_steps = math.floor(num_valid_samples/BATCH_SIZE)

    # gen = keras.preprocessing.image.ImageDataGenerator()
    gen = keras.preprocessing.image.ImageDataGenerator(horizontal_flip=True)
    val_gen = keras.preprocessing.image.ImageDataGenerator(horizontal_flip=True, vertical_flip=True)

    batches = gen.flow_from_directory(TRAIN_DIR, color_mode='rgb', target_size=SIZE, class_mode='categorical', shuffle=True, batch_size=BATCH_SIZE)
    val_batches = val_gen.flow_from_directory(VALID_DIR, color_mode='rgb', target_size=SIZE, class_mode='categorical', shuffle=True, batch_size=BATCH_SIZE)

    base_model = keras.applications.resnet50.ResNet50(input_shape=INPUT_SHAPE, include_top=False)

    classes = list(iter(batches.class_indices))
    base_model.layers.pop()
    for layer in base_model.layers:
        layer.trainable=False
    last = base_model.layers[-1].output

    x = Flatten()(last)
    # and a logistic layer -- let's say we have 200 classes
    x = Dense(len(classes), activation="softmax")(x)
    # x = Dense(len(classes), activation="softmax")(last)
    finetuned_model = Model(base_model.input, x)
    finetuned_model.summary()

    finetuned_model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    for c in batches.class_indices:
        classes[batches.class_indices[c]] = c
    finetuned_model.classes = classes

    early_stopping = EarlyStopping(patience=10)
    checkpointer = ModelCheckpoint('resnet50_best.h5', verbose=1, save_best_only=True)

    history = finetuned_model.fit_generator(batches, steps_per_epoch=num_train_steps, epochs=EPOCHS, callbacks=[early_stopping, checkpointer], validation_data=val_batches, validation_steps=num_valid_steps)
    finetuned_model.save('resnet50_final.h5')
    score = finetuned_model.evaluate_generator(val_batches, steps=num_valid_steps)

    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    print(history.history.keys())

    print('Training loss:', history.history['loss'][-1])
    print('Training accuracy:', history.history['acc'][-1])

    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='best')
    plt.title('Model Accuracy')
    plt.savefig('model_acc.png')
    plt.figure()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='best')
    plt.title('Model Loss')
    plt.savefig('model_loss.png')