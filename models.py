import tensorflow as tf
from tensorflow.keras import layers


def xception_original_model(input_shape=(228, 228, 3),
                              filters=64,
                              num_classes=3,
                              ):
    version = 2
    base_model = tf.keras.applications.Xception(include_top=False,
                                                  weights=None,
                                                  input_tensor=None,
                                                  input_shape=input_shape,
                                                  pooling="max",
                                                  classes=num_classes,
                                                  classifier_activation="softmax",
                                                  )
    base_model.trainable = True
    x = layers.BatchNormalization(momentum=0.9)(base_model.output)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.5)(x)  # <= added dropout layer [ ]
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x_out = layers.Dense(num_classes, activation='softmax')(x)
    keras_model = tf.keras.models.Model(inputs=base_model.input, outputs=x_out)
    name_of_model = f"Xception_orig_{version}_{input_shape[0]}x{input_shape[1]}"
    return keras_model, name_of_model

