
def get_mnist_data():
    (Xtrain, ytrain), (Xtest, ytest) = mnist.load_data()
    return Xtrain, ytrain, Xtest, ytest

def prepare_data(Xtrain, ytrain, Xtest, ytest):
    Xtrain = np.reshape(Xtrain, [-1, IMG_SIZE, IMG_SIZE, IMG_CHAN])
    Xtrain = X_train.astype(np.float32) / 255
    Xtest = np.reshape(Xtest, [-1, IMG_SIZE, IMG_SIZE, IMG_CHAN])
    Xtest = Xtest.astype(np.float32) / 255
    to_categorical = tf.keras.utils.to_categorical
    ytrain = to_categorical(ytrain)
    ytest = to_categorical(ytest)
    return Xtrain, ytrain, Xtest, ytest

def split_data(Xtrain, ytrain):
    idx = np.random.permutation(Xtrain.shape[0])
    Xtrain, ytrain = Xtrain[idx], ytrain[idx]
    n = int(Xtrain.shape[0] * (1 - VALIDATION_SPLIT))
    Xvalid = Xtrain[n:]
    Xtrain = Xtrain[:n]
    yvalid = ytrain[n:]
    ytrain = ytrain[:n]
    return Xtrain, ytrain, Xvalid,  yvalid