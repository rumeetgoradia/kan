import tensorflow as tf

from .r2 import ThreeDimensionalR2Score


def compile_model(model, learning_rate):
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0)
    model.compile(optimizer=optimizer, loss=tf.keras.losses.MeanSquaredError(),
                  metrics=[tf.keras.metrics.MeanAbsoluteError(name='mae'),
                           tf.keras.metrics.MeanSquaredError(name='mse'),
                           tf.keras.metrics.RootMeanSquaredError(name='rmse'),
                           ThreeDimensionalR2Score(name='r2')])

    return model
