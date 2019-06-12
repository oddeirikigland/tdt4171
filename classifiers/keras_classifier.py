import tensorflow as tf
import os
import pickle
from tensorflow import keras

# print(tf.test.gpu_device_name())


def get_data(filename):
    data = pickle.load(open(filename, "rb"))
    x_train, x_test, y_train, y_test, max_length, vocab_size = (
        data["x_train"],
        data["x_test"],
        data["y_train"],
        data["y_test"],
        data["max_length"],
        data["vocab_size"],
    )
    x_train_transform = keras.preprocessing.sequence.pad_sequences(
        x_train, maxlen=max_length
    )
    x_test_transform = keras.preprocessing.sequence.pad_sequences(
        x_test, maxlen=max_length
    )
    y_train_categorized = keras.utils.to_categorical(y_train, num_classes=2)
    y_test_categorized = keras.utils.to_categorical(y_test, num_classes=2)
    return (
        x_train_transform,
        x_test_transform,
        y_train_categorized,
        y_test_categorized,
        vocab_size,
    )


def make_fit_model(x_train, y_train, vocab_size):
    model = keras.Sequential()

    model.add(keras.layers.Embedding(input_dim=vocab_size, output_dim=16))
    model.add(keras.layers.LSTM(units=16))
    model.add(keras.layers.Dense(units=2))

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    # with tf.device('/GPU:0'):
    history = model.fit(x=x_train, y=y_train, batch_size=256, epochs=6, verbose=0)
    model.save("lstm.h5")
    return model, history


def get_full_path(rel_path):
    return os.path.join(os.path.dirname(__file__), rel_path)


def main():
    keras_data = "data/keras-data.pickle"
    x_train_transform, x_test_transform, y_train_categorized, y_test_categorized, vocab_size = get_data(
        get_full_path(keras_data)
    )

    fitted_model, history = make_fit_model(
        x_train_transform, y_train_categorized, vocab_size
    )
    # fitted_model = keras.models.load_model(get_full_path("lstm.h5"))
    loss, acc = fitted_model.evaluate(x_test_transform, y_test_categorized, verbose=0)

    print("loss: {}".format(loss))
    print("Accuracy: {}".format(acc))


if __name__ == "__main__":
    main()
