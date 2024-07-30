import matplotlib.pyplot as plt
import torch

def plot_loss_epoch(train_loss_history, test_loss_history):
    plt.plot(train_loss_history, label='Train')
    plt.plot(test_loss_history, label='Test')
    plt.legend()
    plt.title('Loss Curve')
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.show()


def plot_price_predictions(pred_train, y_train, pred_test, y_test):
    plt.close()
    plt.plot(y_train, label="True Close Price")
    plt.plot(pred_train, label="Predication")
    plt.title("Training Data")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.show()
    
    

    plt.close()
    plt.plot(y_test, label="True Close Price")
    plt.plot(pred_test, label="Predication")
    plt.title("Testing Data")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.show()
