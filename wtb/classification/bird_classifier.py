import os

import joblib
import numpy as np
from PIL import Image
from keras import applications
from keras.applications.vgg16 import preprocess_input
from keras.callbacks import ModelCheckpoint
from keras.layers import Flatten, Dropout, Dense
from keras.models import Model, Sequential
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
from loguru import logger

IMG_ROWS = 240
IMG_COLS = 240
EPOCHS = 10  # Number of times each sample is given to the network
BATCH_SIZE = 32  # Numer of samples given to the network before updating the model
NUM_OF_TRAIN_SAMPLES = 3000
NUM_OF_TEST_SAMPLES = 600
ARCHIVE_FORMAT = "gzip"


class BirdClassifier:
    def __init__(self):
        self.parameters = {
            "input_shape": (IMG_ROWS, IMG_COLS, 3),
            "num_of_classes": 7,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE
        }

        # As the last DL layer depends on the number of classes to predict, the classifier is initialised at the fit
        # step, and thus matches the number of classes (dynamically set from the number of subfolder)
        self.classifier = Model()

        # Map labels to prediction class. Built when training the network.
        self.mapper = {}

    def get_human_prediction(self, prediction: np.array) -> tuple:
        best_proba_index = np.argmax(prediction, axis=1)[0]
        return self.mapper.get(best_proba_index), prediction[0][best_proba_index] * 100

    @staticmethod
    def _build_classifier(num_of_classes: int) -> Model:
        # Load VGG16 model from keras
        base_model = applications.VGG16(
            weights="imagenet", include_top=False, input_shape=(IMG_ROWS, IMG_COLS, 3)
        )

        # Prevent the VGG16 layers to be modified by our custom training
        for layer in base_model.layers:
            layer.trainable = False

        # Build our own top layers for prediction
        model_ft_top = Sequential()
        model_ft_top.add(Flatten())
        model_ft_top.add(Dense(1024, activation="relu"))
        model_ft_top.add(Dropout(0.5))
        model_ft_top.add(Dense(num_of_classes, activation="softmax"))

        # Merge the VGG16 and our custom models
        model_ft = Model(inputs=base_model.input, outputs=model_ft_top(base_model.output))

        # model_ft = Model(inputs=Tensor(base_model.input), outputs=Tensor(model_ft_top(base_model.output)))

        model_ft.compile(
            optimizer=SGD(lr=1e-4, momentum=0.9),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model_ft

    def fit(self, data_train_dir: str, data_valid_dir: str, model_dir: str):
        # Generator for train
        train_image_generator = ImageDataGenerator(preprocessing_function=applications.vgg16.preprocess_input)
        train_iterator = train_image_generator.flow_from_directory(
            data_train_dir,  # Root directory
            target_size=(IMG_ROWS, IMG_COLS),  # Images will be processed to this size
            batch_size=BATCH_SIZE,  # How many data are processed at the same time ?
            class_mode="categorical",
        )  # Each subdir is a category

        # Generator for validation
        valid_image_generator = ImageDataGenerator(preprocessing_function=applications.vgg16.preprocess_input)
        valid_iterator = valid_image_generator.flow_from_directory(
            data_valid_dir,
            target_size=(IMG_ROWS, IMG_COLS),  # Images will be processed to this size
            batch_size=BATCH_SIZE,  # How many data are processed at the same time ?
            class_mode="categorical",
        )

        num_of_classes = len(train_iterator.class_indices)

        # This Keras Callbak saves the best model according to the accuracy metric
        # filepath = os.path.join(model_dir, "{epoch:02d}-{val_accuracy:.2f}.hdf5")
        filepath = os.path.join(model_dir, "best_model.hdf5")
        checkpoint = ModelCheckpoint(
            filepath, monitor="val_accuracy", verbose=1, save_best_only=True, mode="max"
        )

        # Build the dict that associates a class to its "human" label (name of the folder)
        self.mapper = {v: k for k, v in train_iterator.class_indices.items()}

        self.classifier = self._build_classifier(num_of_classes)

        self.classifier.fit_generator(
            generator=train_iterator,
            steps_per_epoch=NUM_OF_TRAIN_SAMPLES // BATCH_SIZE,
            epochs=EPOCHS,
            validation_data=valid_iterator,
            validation_steps=NUM_OF_TEST_SAMPLES // BATCH_SIZE,
            callbacks=[checkpoint],
        )

        # Load the best encountered model weights
        self.classifier.load_weights(filepath)

    def predict(self, image_path: str):
        picture = Image.open(image_path)
        picture = picture.resize(size=(IMG_ROWS, IMG_COLS))
        picture_array = img_to_array(img=picture)
        picture_array = np.expand_dims(picture_array, axis=0)
        prediction = self.classifier.predict(preprocess_input(picture_array))

        logger.info(prediction)
        return prediction

    @staticmethod
    def load_classifier(bird_classifier_filepath: str) -> "BirdClassifier":
        logger.info(f"Load classifier from {bird_classifier_filepath}")
        return joblib.load(bird_classifier_filepath)

    def save(self, file_path: str) -> None:
        archive = f"{file_path}.{ARCHIVE_FORMAT}"
        logger.info(f"Save the classifier {archive}")
        joblib.dump(self, f"{archive}", compress=ARCHIVE_FORMAT)
