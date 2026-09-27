from models.cnn import build_simple_cnn_binary
from models.cnn_cbam import build_cbam_cnn_binary
from models.cnn_transformer import build_cnn_transformer_binary
from models.cnn_cbam_transformer import build_cnn_cbam_transformer_binary


BUILDERS = {
    "cnn": build_simple_cnn_binary,
    "cnn_cbam": build_cbam_cnn_binary,
    "cnn_transformer": build_cnn_transformer_binary,
    "cnn_cbam_transformer": build_cnn_cbam_transformer_binary,
}


def build_model(name, image_size):
    return BUILDERS[name](tuple(image_size))
