import os
import tarfile
from random import randrange

import numpy as np
import numpy.typing as npt
from imgaug import augmenters as iaa
from PIL import Image

DATA_DIR = os.path.join(os.path.dirname(os.path.relpath(__file__)), "data")
SHAPES_DIR = os.path.join(DATA_DIR, "shapes")
TAR_PATH = os.path.join(DATA_DIR, "shapes.tar.gz")
CHANNELS_SHUFFLE = (
    (0, 1, 2),
    (0, 2, 1),
    (1, 0, 2),
    (1, 2, 0),
    (2, 0, 1),
    (2, 1, 0)
)
CHANNELS_INVERT = (
    (0, 0, 0),
    (0, 0, 1),
    (0, 1, 0),
    (0, 1, 1),
    (1, 0, 0),
    (1, 0, 1),
    (1, 1, 0),
    (1, 1, 1)
)
seq = iaa.Sequential([
    iaa.Affine(
        scale=(0.9375, 1.0625),
        rotate=(-360, 360),
        mode="edge"
    ),
    iaa.AddToBrightness(),
    iaa.AddToHue((-75, +75)),
    iaa.AddToSaturation(),
    iaa.Resize(64, interpolation="linear")
])
resize = iaa.Resize(64, interpolation="linear")


class Captcha:
    def __init__(self, img_path: str) -> None:
        img = Image.open(img_path)
        self.__shapes: list[npt.NDArray[np.uint8]] = []

        # first chape
        shape = img.crop((0, 0, 50, 50))
        self.__shapes.append(np.asarray(shape))

        # remaining shapes
        img = img.crop((0, 50, img.width, img.height))

        for row in range(img.height // 50):
            for col in range(img.width // 50):
                left = col * 50
                upper = row * 50
                right = left + 50
                lower = upper + 50
                shape = img.crop((left, upper, right, lower))
                self.__shapes.append(np.asarray(shape))

    @property
    def shapes(self) -> list[npt.NDArray[np.uint8]]:
        return self.__shapes

    @property
    def x_test(self) -> npt.NDArray[np.float32]:
        shapes = np.array(self.__shapes, dtype=np.uint8)
        return (resize(images=shapes) / 255.0).astype(np.float32)


class Dataset:
    def __init__(self) -> None:
        if os.path.exists(SHAPES_DIR):
            self.__load_shapes()
        else:
            self.__create_shapes()

    def __load_shapes(self) -> None:
        self.__x_train: npt.NDArray[np.uint8] = np.load(
            os.path.join(DATA_DIR, "x_train.npy"))
        self.__y_train: npt.NDArray[np.uint8] = np.load(
            os.path.join(DATA_DIR, "y_train.npy"))
        self.__x_test: npt.NDArray[np.uint8] = np.load(
            os.path.join(DATA_DIR, "x_test.npy"))
        self.__y_test: npt.NDArray[np.uint8] = np.load(
            os.path.join(DATA_DIR, "y_test.npy"))

    def __save_shapes(self) -> None:
        np.save(os.path.join(DATA_DIR, "x_train.npy"), self.__x_train)
        np.save(os.path.join(DATA_DIR, "y_train.npy"), self.__y_train)
        np.save(os.path.join(DATA_DIR, "x_test.npy"), self.__x_test)
        np.save(os.path.join(DATA_DIR, "y_test.npy"), self.__y_test)

    def __create_shapes(self) -> None:
        with tarfile.open(TAR_PATH) as tf:
            tf.extractall(DATA_DIR)

        entries = [os.path.join(SHAPES_DIR, entry.split(".")[0])
                   for entry in sorted(os.listdir(SHAPES_DIR))
                   if entry.endswith(".png")]
        total = len(entries)
        valid = total // 8
        train = total - valid
        self.__x_train, self.__y_train = self.__generate_data(entries[:train])
        self.__x_test, self.__y_test = self.__generate_data(entries[train:])
        self.__save_shapes()

    def __generate_data(self, entries: list[str]) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8]]:
        shapes = []
        labels = []

        for entry in entries:
            # shapes
            shapes.extend(Captcha(f"{entry}.png").shapes)

            # labels
            with open(f"{entry}.txt") as mf:
                labels.extend([[l[0]] for l in mf.readlines()])

        return np.array(shapes, dtype=np.uint8), np.array(labels, dtype=np.uint8)

    def __augment_data(self, x_data: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
        aug_data = x_data.copy()

        # shuffle & invert channels randomly
        for s in range(aug_data.shape[0]):
            i = randrange(48)
            si, ii = divmod(i, 8)

            for c in range(3):
                channel = x_data[s, :, :, CHANNELS_SHUFFLE[si][c]]
                aug_data[s, :, :, c] = 255 - channel \
                    if CHANNELS_INVERT[ii][c] \
                    else channel

        return seq(images=aug_data)

    @property
    def x_train(self) -> npt.NDArray[np.float32]:
        return (resize(images=self.__x_train) / 255.0).astype(np.float32)

    @property
    def y_train(self) -> npt.NDArray[np.uint8]:
        return self.__y_train

    @property
    def x_train_augmented(self) -> npt.NDArray[np.float32]:
        return (self.__augment_data(self.__x_train) / 255.00).astype(np.float32)

    @property
    def x_test(self) -> npt.NDArray[np.float32]:
        return (resize(images=self.__x_test) / 255.0).astype(np.float32)

    @property
    def y_test(self) -> npt.NDArray[np.uint8]:
        return self.__y_test
