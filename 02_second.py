import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
import benya_helper_functions

def everything():
    (train_data, train_labels), (test_data, test_labels) = fashion_mnist.load_data()
    # make a list of class names
    class_names = ["T-shirt", "Trouser", "pullover", "dress", "coat", "sandal", "shirt", "sneaker", "bag", "ankle boot"]

    # normalize the training and validation data
    train_data_norm = train_data / 255.0
    test_data_norm = test_data / 255.0

    model(train_data_norm, train_labels, test_data_norm, test_labels)
    print("file finished")





def model(train_data_norm, train_labels, test_data_norm, test_labels):
    # set random seed
    tf.random.set_seed(42)

    # build the model
    model_14 = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(4, activation='relu'),
        tf.keras.layers.Dense(4, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    # compile the model
    model_14.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                     optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                     metrics=['accuracy'])

    model_14_history = model_14.fit(train_data_norm,
                                    train_labels,
                                    epochs=10,  # this was 40
                                    validation_data=(test_data_norm, test_labels))
    print("end model_14")


if __name__ == '__main__':
    everything()
    benya_helper_functions.make_confusion_matrix(y_true=test_labels,
                    y_pred=y_preds,
                    classes=class_names,
                    figsize=(15,15),
                    text_size=10)

