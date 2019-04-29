import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_curve, auc

# plots
# train and test accuracy plot
def plot_accuracy(model, title,save_flag = False):
    train_accu = pd.Series(model.history['acc'])
    print("Mean training accuracy: %.2f" % (train_accu.mean()))
    test_accu = pd.Series(model.history['val_acc'])
    print("Mean testing accuracy: %.2f" % (test_accu.mean()))

    plt.plot(model.history['acc'])
    plt.plot(model.history['val_acc'])
    plt.title('%s:model accuracy' % (title))
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='lower right')
    plt.savefig("%s.png" % (title))
    if save_flag:
        plt.savefig("./accuracy.png")
    plt.show()


# train and test loss plot
def plot_loss(model, title,save_flag = False):
    train_accu = pd.Series(model.history['loss'])
    print("Mean training loss: %.2f" % (train_accu.mean()))
    test_accu = pd.Series(model.history['val_loss'])
    print("Mean testing loss: %.2f" % (test_accu.mean()))

    plt.plot(model.history['loss'])
    plt.plot(model.history['val_loss'])
    plt.title('%s: model loss' % (title))
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    if save_flag:
        plt.savefig("./loss.png")

    plt.show()

# accuracy_plot(keras_class, 'KerasClassifier model')
# loss_plot(keras_class, 'KerasClassifier model')
import itertools
import numpy as np
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(y_true, y_pred, num_classes, title = "Confusion matrix",
                          cmap = plt.cm.Blues, save_flag = False):
    classes = [str(i) for i in range(num_classes)]
    labels = range(num_classes)

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(14, 12))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=40)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, fontsize=20)
    plt.yticks(tick_marks, classes, fontsize=20)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label', fontsize=30)
    plt.xlabel('Predicted label', fontsize=30)

    if save_flag:
        plt.savefig("./confusion_matrix.png")

    plt.show()

def plot_roc(y_true, y_pred):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    print("Area under the ROC curve : %f" % roc_auc)
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()