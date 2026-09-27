from tensorflow.keras import Input, Model
from tensorflow.keras.layers import (
    Conv2D, BatchNormalization, MaxPooling2D, GlobalAveragePooling2D,
    GlobalMaxPooling2D, Reshape, Dense, multiply, add, Activation,
    Lambda, Concatenate, Dropout
)
import tensorflow as tf

# CBAM block definition
def cbam_block(feature_map, ratio=8):
    channel = feature_map.shape[-1]

    shared_dense_one = Dense(channel // ratio, activation='relu', kernel_initializer='he_normal', use_bias=True)
    shared_dense_two = Dense(channel, kernel_initializer='he_normal', use_bias=True)

    avg_pool = GlobalAveragePooling2D()(feature_map)
    avg_pool = Reshape((1, 1, channel))(avg_pool)
    avg_pool = shared_dense_one(avg_pool)
    avg_pool = shared_dense_two(avg_pool)

    max_pool = GlobalMaxPooling2D()(feature_map)
    max_pool = Reshape((1, 1, channel))(max_pool)
    max_pool = shared_dense_one(max_pool)
    max_pool = shared_dense_two(max_pool)

    channel_attention = add([avg_pool, max_pool])
    channel_attention = Activation('sigmoid')(channel_attention)
    channel_refined = multiply([feature_map, channel_attention])

    avg_pool_spatial = Lambda(lambda x: tf.reduce_mean(x, axis=-1, keepdims=True))(channel_refined)
    max_pool_spatial = Lambda(lambda x: tf.reduce_max(x, axis=-1, keepdims=True))(channel_refined)
    concat = Concatenate(axis=-1)([avg_pool_spatial, max_pool_spatial])

    spatial_attention = Conv2D(1, (7, 7), padding='same', activation='sigmoid', kernel_initializer='he_normal')(concat)
    refined_feature = multiply([channel_refined, spatial_attention])

    return refined_feature

# CBAM CNN model
def build_cbam_cnn_binary(input_shape):
    inputs = Input(shape=input_shape)

    # Block 1
    x = Conv2D(32, (3,3), activation='relu', padding='same')(inputs)
    x = BatchNormalization()(x)
    x = cbam_block(x)
    x = MaxPooling2D((2,2))(x)

    # Block 2
    x = Conv2D(64, (3,3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = cbam_block(x)
    x = MaxPooling2D((2,2))(x)

    # Block 3
    x = Conv2D(128, (3,3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = cbam_block(x)
    x = MaxPooling2D((2,2))(x)

    # Block 4
    x = Conv2D(256, (3,3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = cbam_block(x)
    x = MaxPooling2D((2,2))(x)

    # Grad-CAM target layer
    x = Conv2D(128, (3,3), activation='relu', padding='same', name='gradcam_conv')(x)

    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)  # Was 64
    x = Dropout(0.4)(x)
    outputs = Dense(1, activation='sigmoid')(x)

    model = Model(inputs, outputs)
    return model