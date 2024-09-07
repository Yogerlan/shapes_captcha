import os

import numpy as np
import tensorflow as tf

from dataset import Captcha, Dataset

MODELS_DIR = os.path.join(os.path.dirname(os.path.relpath(__file__)), "models")


class Classifier:
    def __init__(
        self,
        model_path=os.path.join(MODELS_DIR, "shapes.keras"),
        best_model_path=os.path.join(MODELS_DIR, "shapes_best.keras"),
        verbose="auto"
    ) -> None:
        self.__model_path = model_path
        self.__best_model_path = best_model_path
        self.__verbose = verbose
        self.__dataset = Dataset()
        self.__best_results = {"accuracy": 0.0, "loss": 100.0}

        if os.path.exists(self.__model_path):
            self.__load_model()
        else:
            self.__create_model()

    def __load_model(self) -> None:
        self.__model = tf.keras.models.load_model(self.__model_path)

        if self.__verbose:
            self.__model.summary()

    def __create_model(self) -> None:
        self.__model = tf.keras.models.Sequential([
            tf.keras.layers.Input((64, 64, 3))
        ])

        for filters in [128, 256, 512]:
            for _ in range(3):
                self.__model.add(tf.keras.layers.Conv2D(
                    filters, 3, padding="same",
                    kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01)))
                self.__model.add(tf.keras.layers.BatchNormalization())
                self.__model.add(tf.keras.layers.LeakyReLU(alpha=0.1))

            if filters != 512:
                self.__model.add(tf.keras.layers.MaxPooling2D())

            self.__model.add(tf.keras.layers.Dropout(0.5))

        self.__model.add(tf.keras.layers.Conv2D(
            5, 1, padding="same",
            kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01)))
        self.__model.add(tf.keras.layers.LeakyReLU(alpha=0.1))
        self.__model.add(tf.keras.layers.GlobalAvgPool2D())
        self.__model.add(tf.keras.layers.Softmax())

        self.__model.compile(
            optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.0001),
            loss="sparse_categorical_crossentropy",
            metrics=["acc"]
        )

        tf.keras.models.save_model(self.__model, self.__model_path)

        if self.__verbose:
            self.__model.summary()

    def load_best_results(self) -> None:
        if os.path.exists(self.__best_model_path):
            best_model = tf.keras.models.load_model(self.__best_model_path)
            self.__best_results = best_model.evaluate(
                self.__dataset.x_test,
                self.__dataset.y_test,
                verbose=self.__verbose,
                return_dict=True
            )

        if self.__verbose:
            print("Best results:", self.__best_results)

    def fit_and_evaluate(self, batch_size=32, epochs=32) -> None:
        for _ in range(epochs):
            self.__model.fit(
                self.__dataset.x_train_augmented,
                self.__dataset.y_train,
                batch_size=batch_size,
                epochs=1
            )
            self.__model.save(self.__model_path)
            results = self.__model.evaluate(
                self.__dataset.x_test,
                self.__dataset.y_test,
                verbose=self.__verbose,
                return_dict=True
            )

            if (self.__best_results["accuracy"] < results["accuracy"]):
                self.__best_results = results
                self.__model.save(self.__best_model_path)

                if self.__verbose:
                    print("Best results:", self.__best_results)

    def predict(self, captcha_path: str) -> list[int]:
        captcha = Captcha(captcha_path)

        return [np.argmax(prediction)
                for prediction in self.__model.predict(captcha.x_test, verbose=self.__verbose)]
