from tensorflow.keras import Input, Model
from tensorflow.keras.layers import (
    Conv2D, BatchNormalization, MaxPooling2D, GlobalAveragePooling2D,
    GlobalMaxPooling2D, Reshape, Dense, multiply, add, Activation,
    Lambda, Concatenate, Dropout, LayerNormalization, Add
)
from tensorflow.keras.layers import MultiHeadAttention
import tensorflow as tf

# CBAM Block
def cbam_block(feature_map, ratio=8):
    channel = feature_map.shape[-1]

    shared_dense_one = Dense(channel // ratio, activation='relu', kernel_initializer='he_normal', use_bias=True)
    shared_dense_two = Dense(channel, kernel_initializer='he_normal', use_bias=True)

    # Channel Attention
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

    # Spatial Attention
    avg_pool_spatial = Lambda(lambda x: tf.reduce_mean(x, axis=-1, keepdims=True))(channel_refined)
    max_pool_spatial = Lambda(lambda x: tf.reduce_max(x, axis=-1, keepdims=True))(channel_refined)
    concat = Concatenate(axis=-1)([avg_pool_spatial, max_pool_spatial])

    spatial_attention = Conv2D(1, (7, 7), padding='same', activation='sigmoid', kernel_initializer='he_normal')(concat)
    refined_feature = multiply([channel_refined, spatial_attention])

    return refined_feature

# Transformer Block
def transformer_block(feature_map, num_heads=4, ff_dim=256, dropout_rate=0.1):
    C = int(feature_map.shape[-1])
    seq = Reshape((-1, C))(feature_map)

    # Self-Attention
    x = LayerNormalization(epsilon=1e-6)(seq)
    attn_out = MultiHeadAttention(num_heads=num_heads, key_dim=C // num_heads)(x, x)
    attn_out = Dropout(dropout_rate)(attn_out)
    x = Add()([seq, attn_out])

    # Feed Forward
    y = LayerNormalization(epsilon=1e-6)(x)
    y = Dense(ff_dim, activation='relu')(y)
    y = Dropout(dropout_rate)(y)
    y = Dense(C)(y)
    y = Dropout(dropout_rate)(y)
    x = Add()([x, y])

    # Reshape back
    H = int(feature_map.shape[1])
    W = int(feature_map.shape[2])
    out_map = Reshape((H, W, C))(x)
    return out_map

# CNN + CBAM + Transformer Model 
def build_cnn_cbam_transformer_binary(input_shape):
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

    # Block 3 + Transformer
    x = Conv2D(128, (3,3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = cbam_block(x)
    x = transformer_block(x, num_heads=4, ff_dim=256, dropout_rate=0.1)
    x = MaxPooling2D((2,2))(x)

    # Block 4 + Transformer
    x = Conv2D(256, (3,3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = cbam_block(x)
    x = transformer_block(x, num_heads=4, ff_dim=256, dropout_rate=0.1)
    x = MaxPooling2D((2,2))(x)

    # Grad-CAM target layer
    x = Conv2D(128, (3,3), activation='relu', padding='same', name='gradcam_conv')(x)

    # Classification Head
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.4)(x)
    outputs = Dense(1, activation='sigmoid')(x)

    model = Model(inputs, outputs)
    return model