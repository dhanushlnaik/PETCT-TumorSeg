from __future__ import annotations

from typing import Optional

import tensorflow as tf
from tensorflow import keras
Model = keras.Model
layers = keras.layers


def conv_block(x: tf.Tensor, filters: int, dropout: float = 0.0, name: Optional[str] = None) -> tf.Tensor:
    """Two consecutive Conv2D + BN + ReLU layers with optional dropout."""
    conv_name = f"{name}_conv" if name else None
    bn_name = f"{name}_bn" if name else None

    x = layers.Conv2D(filters, 3, padding="same", kernel_initializer="he_normal", name=conv_name)(x)
    x = layers.BatchNormalization(name=bn_name)(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(filters, 3, padding="same", kernel_initializer="he_normal", name=None if name is None else f"{name}_conv2")(x)
    x = layers.BatchNormalization(name=None if name is None else f"{name}_bn2")(x)
    x = layers.Activation("relu")(x)

    if dropout > 0.0:
        x = layers.Dropout(dropout)(x)
    return x


def encoder_block(x: tf.Tensor, filters: int, dropout: float = 0.0) -> tuple[tf.Tensor, tf.Tensor]:
    features = conv_block(x, filters, dropout=dropout)
    downsampled = layers.MaxPooling2D(pool_size=(2, 2))(features)
    return features, downsampled


def decoder_block(x: tf.Tensor, skip: tf.Tensor, filters: int) -> tf.Tensor:
    x = layers.Conv2DTranspose(filters, kernel_size=2, strides=2, padding="same")(x)
    x = layers.Concatenate()([x, skip])
    x = conv_block(x, filters)
    return x


def build_unet(
    input_shape: tuple[int, int, int] = (256, 256, 2),
    base_filters: int = 32,
    dropout: float = 0.1,
    num_classes: int = 1,
) -> Model:
    """Construct a 2D U-Net model."""
    inputs = layers.Input(shape=input_shape)

    s1, p1 = encoder_block(inputs, base_filters, dropout=dropout)
    s2, p2 = encoder_block(p1, base_filters * 2, dropout=dropout)
    s3, p3 = encoder_block(p2, base_filters * 4, dropout=dropout)
    s4, p4 = encoder_block(p3, base_filters * 8, dropout=dropout)

    bottleneck = conv_block(p4, base_filters * 16, dropout=dropout)

    d1 = decoder_block(bottleneck, s4, base_filters * 8)
    d2 = decoder_block(d1, s3, base_filters * 4)
    d3 = decoder_block(d2, s2, base_filters * 2)
    d4 = decoder_block(d3, s1, base_filters)

    outputs = layers.Conv2D(num_classes, kernel_size=1, activation="sigmoid")(d4)

    model = Model(inputs=inputs, outputs=outputs, name="unet")
    return model


def dice_coefficient(y_true: tf.Tensor, y_pred: tf.Tensor, smooth: float = 1e-6) -> tf.Tensor:
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)


def dice_loss(y_true: tf.Tensor, y_pred: tf.Tensor, smooth: float = 1e-6) -> tf.Tensor:
    return 1.0 - dice_coefficient(y_true, y_pred, smooth)
