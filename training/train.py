import tensorflow as tf
from tensorflow.keras import callbacks


def train_model(model, train_ds, val_ds, config, model_name):
    settings = config["models"][model_name]
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=config["learning_rate"]
        ),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    early_stop = callbacks.EarlyStopping(
        monitor="val_loss",
        patience=settings["early_stopping_patience"],
        restore_best_weights=True,
    )
    lr_scheduler = callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=settings["lr_factor"],
        patience=settings["lr_patience"],
    )
    return model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config["epochs"],
        callbacks=[early_stop, lr_scheduler],
    )
