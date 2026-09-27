from tensorflow.keras import Input, Model
from tensorflow.keras.layers import (
    Conv2D, BatchNormalization, MaxPooling2D, GlobalAveragePooling2D,
    Dense, Dropout, LayerNormalization, Add, Reshape
)
from tensorflow.keras.layers import MultiHeadAttention
import tensorflow as tf

# Transformer Encoder Block (safe: use only when HxW is reasonably small)
def transformer_block(feature_map, num_heads=4, ff_dim=256, dropout_rate=0.1):
    """
    feature_map: 4D tensor (batch, H, W, C) with modest H*W (e.g., 28x28 or 14x14)
    returns: 4D tensor with same shape
    """
    # Channels
    C = int(feature_map.shape[-1])

    # Flatten spatial dims -> (batch, N, C) where N = H*W
    seq = Reshape((-1, C))(feature_map)

    # Pre-norm
    x = LayerNormalization(epsilon=1e-6)(seq)

    # Multi-head self-attention
    attn_out = MultiHeadAttention(num_heads=num_heads, key_dim=C // num_heads)(x, x)
    attn_out = Dropout(dropout_rate)(attn_out)
    x = Add()([seq, attn_out])  # Residual

    # Feed-forward
    y = LayerNormalization(epsilon=1e-6)(x)
    y = Dense(ff_dim, activation='relu')(y)
    y = Dropout(dropout_rate)(y)
    y = Dense(C)(y)
    y = Dropout(dropout_rate)(y)
    x = Add()([x, y])  # Residual

    # Reshape back to (batch, H, W, C)
    H = int(feature_map.shape[1])
    W = int(feature_map.shape[2])
    out_map = Reshape((H, W, C))(x)
    return out_map

# CNN + Transformer model (transformers only after Block 3 & Block 4)
def build_cnn_transformer_binary(input_shape):
    inputs = Input(shape=input_shape)

    # Block 1
    x = Conv2D(32, (3,3), activation='relu', padding='same')(inputs)
    x = BatchNormalization()(x)
    # no transformer here (too large)
    x = MaxPooling2D((2,2))(x)

    # Block 2
    x = Conv2D(64, (3,3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    # still avoid transformer here
    x = MaxPooling2D((2,2))(x)

    # Block 3
    x = Conv2D(128, (3,3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    # Transformer here: spatial size is reduced (e.g., 28x28 for 224 input)
    x = transformer_block(x, num_heads=4, ff_dim=256, dropout_rate=0.1)
    x = MaxPooling2D((2,2))(x)

    # Block 4
    x = Conv2D(256, (3,3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    # Another transformer at even smaller spatial resolution (e.g., 14x14)
    x = transformer_block(x, num_heads=4, ff_dim=256, dropout_rate=0.1)
    x = MaxPooling2D((2,2))(x)

    # Grad-CAM target layer (unchanged)
    x = Conv2D(128, (3,3), activation='relu', padding='same', name='gradcam_conv')(x)

    # Classification Head
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.4)(x)
    outputs = Dense(1, activation='sigmoid')(x)

    model = Model(inputs, outputs)
    return model