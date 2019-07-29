import os
import sys
import pdb
import argparse
import utils
import numpy as np
from glob import glob
from collections import defaultdict
from random import choice, sample
import pandas as pd
import resnet
from tensorflow.keras.layers import Input, Dense, GlobalMaxPool2D, GlobalAvgPool2D, Concatenate, Multiply, Dropout, Subtract
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import image

sys.path.append('../tool')
import toolkits

weight_file = "../model/resnet50_softmax_dim512/weights.h5"
if len(sys.argv) > 1:
    weight_file = sys.argv[1] #"./vgg_face2_0.h5"
train_file_path = "../../recognizing-faces-in-the-wild/train_relationships.csv"
train_folders_path = "../../recognizing-faces-in-the-wild/train/"
val_famillies_list = ["F07", "F08", "F09"]

all_images = glob(train_folders_path + "*/*/*.jpg")
relationships = pd.read_csv(train_file_path)


def initialize_model():
    from model import Vggface2_ResNet50
    # Set basic environments.
    # Initialize GPUs
    toolkits.initialize_GPU()

    # ==> loading the pre-trained model.
    input1 = Input(shape=(224, 224, 3))
    input2 = Input(shape=(224, 224, 3))
    # x1 = resnet.resnet50_backend(input1)
    # x2 = resnet.resnet50_backend(input2)
    base_model = Vggface2_ResNet50(include_top=False)
    base_model.load_weights(weight_file, by_name=True)
    print("successfully load model ", weight_file)
    for x in base_model.layers:
        x.trainable = True
    x1 = base_model(input1)
    x2 = base_model(input2)

    x1 = Concatenate(axis=-1)([GlobalMaxPool2D()(x1), GlobalAvgPool2D()(x1)])
    x2 = Concatenate(axis=-1)([GlobalMaxPool2D()(x2), GlobalAvgPool2D()(x2)])

    x3 = Subtract()([x1, x2])
    x3 = Multiply()([x3, x3])

    x1_ = Multiply()([x1, x1])
    x2_ = Multiply()([x2, x2])
    x4 = Subtract()([x1_, x2_])
    x = Concatenate(axis=-1)([x4, x3])

    x = Dense(100, activation="relu")(x)
    x = Dropout(0.3)(x)
    x = Dense(25, activation="relu")(x)
    x = Dropout(0.3)(x)
    out = Dense(1, activation="sigmoid")(x)

    model = Model([input1, input2], out)
    # for x in model.layers[-21:]:
    #     x.trainable = True

    model.compile(loss="binary_crossentropy", metrics=['acc'], optimizer=Adam(0.00005))

    model.summary()

    return model


def get_train_val(family_name, relationships=relationships):
    # Get val_person_image_map
    val_famillies = family_name
    train_images = [x for x in all_images if val_famillies not in x]
    val_images = [x for x in all_images if val_famillies in x]

    train_person_to_images_map = defaultdict(list)

    ppl = [x.split("/")[-3] + "/" + x.split("/")[-2] for x in all_images]

    for x in train_images:
        train_person_to_images_map[x.split("/")[-3] + "/" + x.split("/")[-2]].append(x)

    val_person_to_images_map = defaultdict(list)

    for x in val_images:
        val_person_to_images_map[x.split("/")[-3] + "/" + x.split("/")[-2]].append(x)

    # Get the train and val dataset
    # relationships = pd.read_csv(train_file_path)
    relationships = list(zip(relationships.p1.values, relationships.p2.values))
    relationships = [x for x in relationships if x[0] in ppl and x[1] in ppl]

    train = [x for x in relationships if val_famillies not in x[0]]
    val = [x for x in relationships if val_famillies in x[0]]

    return train, val, train_person_to_images_map, val_person_to_images_map

def read_img(path):
    img = image.load_img(path) #, target_size=(197, 197))
    img = np.array(img).astype(np.float)
    if img.shape != (224,224,3):
        raise IOError
    return preprocess_input(img, version=2)

def gen(list_tuples, person_to_images_map, batch_size=16):
    ppl = list(person_to_images_map.keys())
    while True:
        batch_tuples = sample(list_tuples, batch_size // 2)
        labels = [1] * len(batch_tuples)
        while len(batch_tuples) < batch_size:
            p1 = choice(ppl)
            p2 = choice(ppl)

            if p1 != p2 and (p1, p2) not in list_tuples and (p2, p1) not in list_tuples:
                batch_tuples.append((p1, p2))
                labels.append(0)

        for x in batch_tuples:
            if not len(person_to_images_map[x[0]]):
                print(x[0])

        X1 = [choice(person_to_images_map[x[0]]) for x in batch_tuples]
        # X1 = np.array([read_img(x) for x in X1])
        X1 = np.array([utils.load_data(x) for x in X1])

        X2 = [choice(person_to_images_map[x[1]]) for x in batch_tuples]
        # X2 = np.array([read_img(x) for x in X2])
        X2 = np.array([utils.load_data(x) for x in X2])

        yield [X1, X2], labels

model1 = initialize_model()
n_val_famillies_list = len(val_famillies_list)

val_acc_list = []
def train_model1():
    for i in range(n_val_famillies_list):
        train, val, train_person_to_images_map, val_person_to_images_map = get_train_val(val_famillies_list[i])
        file_path = "vgg_face2_{}.h5".format(i)
        checkpoint = ModelCheckpoint(file_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        reduce_on_plateau = ReduceLROnPlateau(monitor="val_acc", mode="max", factor=0.1, patience=10, verbose=1)
        es = EarlyStopping(monitor="val_acc", min_delta = 0.001, patience=20, verbose=1)
        callbacks_list = [checkpoint, reduce_on_plateau, es]

        history = model1.fit_generator(gen(train, train_person_to_images_map, batch_size=16),
                                      use_multiprocessing=True,
                                      validation_data=gen(val, val_person_to_images_map, batch_size=16),
                                      epochs=100, verbose=0,
                                      workers=4, callbacks=callbacks_list,
                                      steps_per_epoch=300, validation_steps=150)
        val_acc_list.append(np.max(history.history['val_acc']))

train_model1()
