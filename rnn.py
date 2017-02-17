import fnmatch
import os
import io
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.contrib import layers


# Data helper functions
def load_shapespeare_txt(directory):
    """Load the text given the input directory.  Data pulled from 
    http://lexically.net/wordsmith/support/shakespeare.html

    Parameters
    ----------
    directory : str
        The directory containing the text

    Returns
    -------
    list of chars
        The list of characters from the text, in order
    """
    # Get list of texts
    text_paths = []
    for sub_directory in os.listdir(directory):
        for file in os.listdir(os.path.join(directory, sub_directory)):
            if fnmatch.fnmatch(file, "*.txt"):
                text_paths.append(os.path.join(directory, sub_directory, file))

    # Load the text from the UTF-16 encoding
    texts = []
    for text_path in text_paths:
        texts.append(io.open(text_path, "r", encoding='utf-16-le').read())

    return " ".join(texts)


def alphabet_to_index(character):
    """Convert an alphabet character to a one hot encoding index.

    Parameters
    ----------
    character : str
        A single character

    Returns
    -------
    int
        The index for that character in a one hot vector
    """
    code = ord(character)
    if code == 9: # \t
        return 1
    elif code == 10: # \n
        return 2
    elif 32 <= code <= 126:
        return code - 29
    else:
        return 0


def index_to_alphabet(index):
    """Convert an index in a one hot encoded vector to a character.

    Parameters
    ----------
    index : int
        The index for that character in a one hot vector

    Returns
    -------
    character : str
        A single character
    """
    if index == 0:
        return ""
    elif index == 1:
        return chr(9)
    elif index == 2:
        return chr(10)
    else:
        return chr(index + 29)


def code_text(text, output_width):
    """Given a list of character and output width, return a list of one hot encoded vectors.

    Parameters
    ----------
    text : list of str
        A list of characters

    Returns
    -------
    List
        A list of encoded character
    """
    return [alphabet_to_index(character) for character in text]


def minibatch_sequencer(data, batch_size, sequence_length):
    """Deliver batches of sequence data.

    Parameters
    ----------
    data : 2d array
        The one hot encoded character sequences
    batch_size : int
        The number of examples to deliver in a batch
    sequence_length : int
        The length of the output sequence
    num_epochs : int
        The number of epochs of data to train

    Returns
    -------
    2d array, 2d array, int
        The X batch, y batch, and current epoch number
    """
    # Calculate the number of batches per epoch
    data = np.array(data)
    n_data = data.shape[0]
    batches_per_epoch = (n_data - 1) / (batch_size * sequence_length) # -1 for output shift
    rounded_n_data = batches_per_epoch * batch_size * sequence_length

    # Get new x and y data
    x = data[0:rounded_n_data].reshape([batch_size, batches_per_epoch * sequence_length])
    y = data[1:(rounded_n_data + 1)].reshape([batch_size, batches_per_epoch * sequence_length])

    # Iterate over bathces
    for batch in range(batches_per_epoch):
        x_batch = x[:, (batch * sequence_length):((batch + 1) * sequence_length)]
        y_batch = y[:, (batch * sequence_length):((batch + 1) * sequence_length)]

        yield x_batch, y_batch


ALPHASIZE=98
def sample_from_probabilities(probabilities, topn=ALPHASIZE):
    """Sample a number given a list of probabilities.

    Parameters
    ----------
    probabilities : np.array
        A list of probabilities
    topn : int
        Number of greatest probabilities to sample from

    Returns
    -------
    int
        A random integer
    """
    p = np.squeeze(probabilities)
    p[np.argsort(p)[:-topn]] = 0
    p = p / np.sum(p)
    return np.random.choice(ALPHASIZE, 1, p=p)[0]


# Constants
ALPHASIZE = 98 # Width of one hot encoded input elements
CELLSIZE = 512 # Size of internal layers
NLAYERS = 3 # Layes of NN
SEQLEN = 30 # Length of sequence input
BATCHSIZE = 100 # Batches of inputs
PKEEP = 1.0

# Create placeholders/variables
batch_size = tf.placeholder(tf.int32)
pkeep = tf.placeholder(tf.float32, name="pkeep")
Xd = tf.placeholder(tf.uint8, [None, None], name="Xd")
X = tf.one_hot(Xd, ALPHASIZE, 1.0, 0.0, name="X")
Yd_ = tf.placeholder(tf.uint8, [None, None], name="Yd_")
Y_ = tf.one_hot(Yd_, ALPHASIZE, 1.0, 0.0, name="Y_")
Hin = tf.placeholder(tf.float32, [None, CELLSIZE * NLAYERS], name="Hin")

# Model
cell = tf.nn.rnn_cell.GRUCell(CELLSIZE)
dropout_cell = tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=pkeep)
multicell = tf.nn.rnn_cell.MultiRNNCell([dropout_cell] * NLAYERS, state_is_tuple=False)
dropout_multicell = tf.nn.rnn_cell.DropoutWrapper(multicell, output_keep_prob=pkeep)
Hr, H = tf.nn.dynamic_rnn(dropout_multicell, X, initial_state=Hin)

# Softmax output layer
Hf = tf.reshape(Hr, [-1, CELLSIZE])
Ylogits = layers.linear(Hf, ALPHASIZE)
Y = tf.nn.softmax(Ylogits)
Yp = tf.argmax(Y, 1)
Yp = tf.reshape(Yp, tf.pack([batch_size, -1]))

# Metrics
cross_entropy = -tf.reduce_mean(tf.reshape(Y_, shape=tf.pack([batch_size * SEQLEN, -1])) * tf.log(tf.clip_by_value(Y, 1e-10, 1.0)))
is_correct = tf.equal(tf.argmax(Y, 1), tf.argmax(tf.reshape(Y_, shape=tf.pack([batch_size * SEQLEN, -1])), 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

# Loss function and optimizer
loss = tf.nn.softmax_cross_entropy_with_logits(Ylogits, Y_)

# Optimizer
train_step = tf.train.AdamOptimizer(1e-3).minimize(loss)

# Load the text and encode it
text = load_shapespeare_txt("shakespeare_text")
coded_text = code_text(text, ALPHASIZE)
coded_train = coded_text[0:int(len(coded_text) * 0.8)]
coded_test = coded_text[int(len(coded_text) * 0.8):]

# Start tf session
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# Train the model
training_accuracies = []
training_cross_entropies = []
test_accuracies = []
test_cross_entropies = []
epochs = 20
num_batches = (len(coded_train) - 1) / (BATCHSIZE * SEQLEN)

# Loop over dataset
for epoch in range(epochs):
    # Keep progress on the batches
    batch_counter = 0

    # Create empty state for first iteration through data
    inH = np.zeros([BATCHSIZE, CELLSIZE * NLAYERS])
    # Pull out batches
    for x, y_ in minibatch_sequencer(coded_train, BATCHSIZE, SEQLEN):
        # Create feed dictionary
        train_dict = {
            batch_size: BATCHSIZE,
            pkeep: PKEEP,
            Xd: x,
            Yd_: y_,
            Hin: inH
        }

        # Run the training step
        _, y, outH = sess.run([train_step, Yp, H], feed_dict=train_dict)

        # Every so often, get the accuracies (this gets messy)
        if batch_counter % 100 == 0:
            # Check the training accuracy
            training_accuracy, training_cross_ent = sess.run([accuracy, cross_entropy],
                                                             feed_dict=train_dict)
            training_accuracies.append(training_accuracy)
            training_cross_entropies.append(training_cross_ent)

            # Check the validation accuracy
            VALID_SEQLEN = 30 # 1024
            VALID_BATCHSIZE = 100 # len(coded_test) / VALID_SEQLEN
            valid_Hin = np.zeros([VALID_BATCHSIZE, CELLSIZE * NLAYERS])
    
            # Unforunately, we have to batch up the validation accurayc
            batch_test_accuracy = []
            batch_test_cross_ent = []
            for valid_x, valid_y in minibatch_sequencer(coded_test, VALID_BATCHSIZE, VALID_SEQLEN):
                valid_dict = {
                    batch_size: VALID_BATCHSIZE,
                    pkeep: 1.0,
                    Xd: valid_x,
                    Yd_: valid_y,
                    Hin: valid_Hin
                }

                test_accuracy, test_cross_ent, valid_Hout = sess.run([accuracy, cross_entropy, H],
                                                                    feed_dict=valid_dict)
                batch_test_accuracy.append(test_accuracy)
                batch_test_cross_ent.append(test_cross_ent)
                valid_Hin = valid_Hout

            # Take the average of those validation batches
            test_accuracies.append(np.mean(batch_test_accuracy))
            test_cross_entropies.append(np.mean(batch_test_cross_ent))

            # Print some statistics
            print "Epoch {}, processed {} of {} batches, {:.2f}% complete!".format(
                epoch,
                batch_counter,
                num_batches,
                float(batch_counter) / num_batches * 100
            )

            print "Epoch {}, training accuracy = {:.2f}%, test accuracy = {:.2f}%!".format(
                epoch,
                training_accuracy * 100,
                test_accuracy * 100
            )

        batch_counter += 1
        inH = outH

    # At each epoch, generate some text to see how we're doing.
    starting_letter = "K"
    x = np.array([[alphabet_to_index(starting_letter)]])
    h = np.zeros([1, CELLSIZE * NLAYERS])
    generated_text = [starting_letter]
    for i in range(1000):
        test_dict = {
            batch_size: 1,
            pkeep: 1.0,
            Xd: x,
            Hin: h
        }

        y, h = sess.run([Y, H], feed_dict=test_dict)
        c = sample_from_probabilities(y, topn=10)
        generated_text.append(index_to_alphabet(c))
        x = np.array([[c]])
    print "Epoch {}\n{}\n\n".format(epoch, "".join(generated_text))


# Generate a longer text
starting_letter = "K"
x = np.array([[alphabet_to_index(starting_letter)]])
h = np.zeros([1, CELLSIZE * NLAYERS])
generated_text = [starting_letter]
for i in range(100000):
    test_dict = {
        batch_size: 1,
        pkeep: 1.0,
        Xd: x,
        Hin: h
    }

    y, h = sess.run([Y, H], feed_dict=test_dict)
    c = sample_from_probabilities(y, topn=10)
    generated_text.append(index_to_alphabet(c))
    x = np.array([[c]])

# Save teh generated text
with open("output/generated_shakespeare.txt", "wb") as out:
    out.write("{}\n\n".format(epoch, "".join(generated_text)))


# Plot the performance
plt.figure(figsize=(15, 8))
plt.subplot(121)
plt.plot(range(0, 10001, 10), training_accuracies)
plt.plot(range(0, 10001, 10), test_accuracies)
plt.ylim(0.94, 1.00)
plt.title("Accuracy")
plt.xlabel("Iterations")
plt.ylabel("Accuracy")
plt.legend(["Training Accuracy", "Test Accuracy"], loc="lower right")
plt.subplot(122)
plt.plot(range(0, 10001, 10), training_cross_entropies)
plt.plot(range(0, 10001, 10), test_cross_entropies)
plt.ylim(0, 20)
plt.title("Cross Entropy Loss")
plt.xlabel("Iterations")
plt.ylabel("Cross Entropy")
plt.legend(["Training Loss", "Test Loss"], loc="upper right")
plt.savefig("output/rnn.png")
