import os
import urllib.request as request
from zipfile import ZipFile
import tensorflow as tf
from pathlib import Path
from cnnClassifier.config.configuration import PrepareBaseModelConfig

#  class PrepareBaseModel that handles the process of setting up a deep learning model for a transfer learning task. The class's methods are responsible for loading a pre-trained model, modifying its architecture for a new classification problem, and saving the models at different stages.
class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):    #  " : " hint that the config parameter should be of type PrepareBaseModelConfig.  
        self.config = config

    #Instance method
    def get_base_model(self):
        self.model = tf.keras.applications.vgg16.VGG16(
            input_shape=self.config.params_image_size,
            weights=self.config.params_weights,
            include_top=self.config.params_include_top
        ) # this line load the VGG16 model with the specified input shape, pre-trained weights, and whether to include the top classification layers.

        self.save_model(path=self.config.base_model_path, model=self.model) #it calls save_MODEL(STATIC MEHTOD) to save the loaded model to the specified path.

    

    @staticmethod
    def _prepare_full_model(model, classes, freeze_all, freeze_till, learning_rate):
        if freeze_all:
            for layer in model.layers:
                model.trainable = False
        elif (freeze_till is not None) and (freeze_till > 0):
            for layer in model.layers[:-freeze_till]:
                model.trainable = False

        flatten_in = tf.keras.layers.Flatten()(model.output)
        prediction = tf.keras.layers.Dense(
            units=classes,
            activation="softmax"
        )(flatten_in)

        full_model = tf.keras.models.Model(
            inputs=model.input,
            outputs=prediction
        )

        full_model.compile(
            optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=["accuracy"]
        )

        full_model.summary()
        return full_model
    
    #instance method
    def update_base_model(self):
        self.full_model = self._prepare_full_model(
            model=self.model,
            classes=self.config.params_classes,
            freeze_all=True,
            freeze_till=None,
            learning_rate=self.config.params_learning_rate
        )

        self.save_model(path=self.config.updated_base_model_path, model=self.full_model)

    
        
    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)

