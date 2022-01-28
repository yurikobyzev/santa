import os
from typing import Tuple
import pytz
import datetime
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import tensorflow_addons as tfa
import seaborn as sns
from models import xception_original_model
from dataset import ImagesDataSet
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

__version__ = 0.006

home_dir = os.getcwd()
base_dir = os.path.join(home_dir, 'data')
train_dir = os.path.join(base_dir, 'train')
weight_dir = os.path.join(base_dir, 'weight')
test_dir = os.path.join(base_dir, 'test')


class TrainNN:
    def __init__(self,
                 dataset: ImagesDataSet,
                 ):
        self.dataset = dataset
        self.y_Pred = None
        self.experiment_name = f"{self.dataset.version}"
        self.history = None
        self.epochs = 20

        """ Use it only if not using TimeSeries Generator"""
        self.batch_size = None
        self.monitor = "categorical_accuracy"
        self.loss = "categorical_crossentropy"
        # self.metric = "categorical_accuracy"
        self.metric = tfa.metrics.F1Score(num_classes=dataset.num_classes)
        self.path_filename: str = ''
        self.model_compiled = False
        self.es_patience = 15
        self.rlrs_patience = 8
        self.keras_model, self.net_name = xception_original_model(input_shape=(self.dataset.image_size,
                                                                                 self.dataset.image_size) + (3,),
                                                                    num_classes=3)

        self.learning_rate = 3e-5
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.class_weights = self.dataset.class_weights

    def compile(self):
        self.path_filename = os.path.join(weight_dir, f"{self.experiment_name}_{self.net_name}_{self.monitor}")
        self.keras_model.summary()
        self.keras_model.compile(optimizer=self.optimizer,
                                 loss=self.loss,
                                 metrics=[self.metric],
                                 )
        self.model_compiled = True
        pass

    def train(self):
        if not self.model_compiled:
            self.compile()
        chkp = ModelCheckpoint(f"{self.path_filename}.h5",
                               mode='auto',
                               monitor=self.monitor,
                               save_best_only=True,
                               )
        rlrs = ReduceLROnPlateau(monitor=self.monitor, factor=0.07, patience=self.rlrs_patience, min_lr=1e-07)
        es = EarlyStopping(patience=self.es_patience, monitor=self.monitor, restore_best_weights=True)
        callbacks = [rlrs, chkp, es]

        path_filename = f"{self.path_filename}_NN.png"

        self.history = self.keras_model.fit(self.dataset.train_gen,
                                            validation_data=self.dataset.val_gen,
                                            epochs=self.epochs,
                                            verbose=1,
                                            callbacks=callbacks,
                                            class_weight=self.class_weights
                                            )
        self.model_compiled = True
        pass

    def load_best_weights(self):
        path_filename = f"{self.path_filename}.h5"
        self.keras_model.load_weights(path_filename)
        pass

    def get_predict(self, x_Data):
        if not self.model_compiled:
            self.compile()
            self.load_best_weights()
        self.y_Pred = self.keras_model.predict(x_Data)
        return self.y_Pred

    def evaluate(self, Test: Tuple):
        if not self.model_compiled:
            self.compile()
            self.load_best_weights()
        self.keras_model.evaluate(Test)
        pass


if __name__ == "__main__":
    start = datetime.datetime.now()
    timezone = pytz.timezone("Europe/Moscow")
    image_size = 224
    batch_size = 8
    epochs = 30
    start_learning_rate = 0.0001
    start_patience = round(epochs * 0.04)

    print(f'Image Size = {image_size}x{image_size}')
    dataset = ImagesDataSet(train_dir,
                            os.path.join(base_dir, "train.csv"),
                            image_size=image_size,
                            )
    dataset.batch_size = batch_size
    dataset.validation_split = 0.1
    dataset.build()
    tr = TrainNN(dataset)
    tr.monitor = "loss"
    tr.learning_rate = start_learning_rate
    tr.es_patience = 20
    tr.rlrs_patience = start_patience
    tr.epochs = epochs
 #  tr.optimizer = tf.keras.optimizers.SGD(learning_rate=tr.learning_rate*10,
 #                                          nesterov=True,
 #                                          momentum=0.9
 #                                          )

    tr.optimizer = tf.keras.optimizers.Adam(learning_rate=tr.learning_rate)

    tr.keras_model, tr.net_name = xception_original_model(input_shape=(tr.dataset.image_size,
                                                                         tr.dataset.image_size) + (3,),
                                                            num_classes=dataset.num_classes)
    tr.train()
    end = datetime.datetime.now()
    print(f'Planned epochs: {epochs} Calculated epochs : {len(tr.history.history["loss"])} Time elapsed: {end - start}')

    """ Checking train on all available data """
    dataset.build_check_gen(batch_size=batch_size)
    tr.evaluate(dataset.all_gen)


    print("ok")
