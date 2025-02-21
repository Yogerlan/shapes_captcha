import os

import keras
import numpy as np
from dataset import Captcha, Dataset

MODELS_DIR = os.path.join(os.path.dirname(os.path.relpath(__file__)), "models")


class CustomPolynomialDecay(keras.optimizers.schedules.LearningRateSchedule):
    def __init__(
        self,
        initial_learning_rate,
        power=1.0,
        max_steps=1,
        name="CustomPolynomialDecay",
    ):
        super().__init__()
        self.initial_learning_rate = keras.ops.convert_to_tensor(
            initial_learning_rate
        )
        dtype = self.initial_learning_rate.dtype
        self.current_learning_rate = self.initial_learning_rate
        self.power = keras.ops.cast(power, dtype)
        self.max_steps = keras.ops.cast(max_steps, dtype)
        self.name = name

    def __call__(self, step):
        with keras.ops.name_scope(self.name):
            current_step = keras.ops.cast(
                step,
                self.initial_learning_rate.dtype
            )
            self.current_learning_rate = keras.ops.multiply(
                self.initial_learning_rate,
                keras.ops.power(
                    1 - keras.ops.divide(
                        current_step,
                        self.max_steps
                    ),
                    self.power
                )
            )

            return self.current_learning_rate

    def get_config(self):
        return {
            "initial_learning_rate": self.initial_learning_rate,
            "current_learning_rate": self.current_learning_rate,
            "power": self.power,
            "max_steps": self.max_steps,
            "name": self.name,
        }


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
        self.__best_results = {"acc": 0.0, "loss": 100.0}

        if os.path.exists(self.__model_path):
            self.__load_model()
        else:
            self.__create_model()

    def __load_model(self) -> None:
        self.__model = keras.models.load_model(self.__model_path)

        if self.__verbose:
            self.__model.summary()

    def __create_model(self) -> None:
        self.__model = keras.models.Sequential([
            keras.layers.Input((64, 64, 3))
        ])

        for filters in [128, 256, 512]:
            for _ in range(3):
                self.__model.add(keras.layers.Conv2D(
                    filters, 3, padding="same",
                    kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.01)))
                self.__model.add(keras.layers.BatchNormalization())
                self.__model.add(keras.layers.LeakyReLU(alpha=0.1))

            if filters != 512:
                self.__model.add(keras.layers.MaxPooling2D())

            self.__model.add(keras.layers.Dropout(0.5))

        self.__model.add(keras.layers.Conv2D(
            5, 1, padding="same",
            kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.01)))
        self.__model.add(keras.layers.LeakyReLU(alpha=0.1))
        self.__model.add(keras.layers.GlobalAvgPool2D())
        self.__model.add(keras.layers.Softmax())

        learning_rate_fn = CustomPolynomialDecay(
            0.1,
            4,
            3762
        )
        self.__model.compile(
            optimizer=keras.optimizers.RMSprop(learning_rate=learning_rate_fn),
            loss="sparse_categorical_crossentropy",
            metrics=["acc"]
        )

        keras.models.save_model(self.__model, self.__model_path)

        if self.__verbose:
            self.__model.summary()

    def load_best_results(self) -> None:
        if os.path.exists(self.__best_model_path):
            best_model = keras.models.load_model(self.__best_model_path)
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

            if (self.__best_results["acc"] < results["acc"]):
                self.__best_results = results
                self.__model.save(self.__best_model_path)

                if self.__verbose:
                    print("Best results:", self.__best_results)

    def predict(self, captcha_path: str) -> list[int]:
        captcha = Captcha(captcha_path)

        return [np.argmax(prediction)
                for prediction in self.__model.predict(captcha.x_test, verbose=self.__verbose)]
