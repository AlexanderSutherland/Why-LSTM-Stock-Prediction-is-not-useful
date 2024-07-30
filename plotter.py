import matplotlib.pyplot as plt

def plot_loss_epoch(train_loss_history, test_loss_history):
    plt.plot(train_loss_history, label='Train')
    plt.plot(test_loss_history, label='Test')
    plt.legend()
    plt.title('Loss Curve')
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.show()


def plot_price_predictions(pred_train, y_train, pred_test, y_test, save = False):
    plt.close()
    plt.plot(y_train, label="True Close Price")
    plt.plot(pred_train, label="Predication Close Price")
    plt.title("Training Data")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    if save:
        plt.savefig('figures/price_pred_train.png') 
    else:
        plt.show()
    

    plt.close()
    plt.plot(y_test, label="True Close Price")
    plt.plot(pred_test, label="Predication Close Price")
    plt.title("Testing Data")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    if save:
        plt.savefig('figures/price_pred_test.png') 
    else:
        plt.show()
    
def plot_daily_change_predictions(pred_train, y_train, pred_test, y_test, save = False):
    plt.close()
    plt.plot(y_train, label="True Daily Price Change")
    plt.plot(pred_train, label="Predication Daily Price Change")
    plt.title("Training Data")
    plt.xlabel("Date")
    plt.ylabel("Daily Change")
    plt.legend()
    if save:
        plt.savefig('figures/daily_change_pred_train.png') 
    else:
        plt.show()
    
    

    plt.close()
    plt.plot(y_test, label="True Daily Price Change")
    plt.plot(pred_test, label="Predication Daily Price Change")
    plt.title("Testing Data")
    plt.xlabel("Date")
    plt.ylabel("Daily Change")
    plt.legend()
    if save:
        plt.savefig('figures/daily_change_pred_test.png') 
    else:
        plt.show()
