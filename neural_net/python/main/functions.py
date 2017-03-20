import keras.backend as K

# calculates mean noise remained on "zero" positions
# approximation for numpy version
# to be Keras parameter
def mean_residual_noise(y_true, y_pred):
    a = 1
    k = 1000000
    return K.mean(K.sum((a / (k * K.pow(y_true, 2) + 1)) * y_pred, axis=1))