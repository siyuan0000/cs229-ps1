import matplotlib.pyplot as plt
import numpy as np
import util


def main(tau, train_path, eval_path):
    """Problem: Locally weighted regression (LWR)

    Args:
        tau: Bandwidth parameter for LWR.
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    # Fit a LWR model
    clf = LocallyWeightedLinearRegression(tau)
    clf.fit(x_train, y_train)
    
    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=True)
    # Get MSE value on the validation set
    y_pred = clf.predict(x_eval)
    mse = np.mean((y_pred - y_eval) ** 2)
    print(f"MSE: {mse}")
    
    # Plot validation predictions on top of training set
    plt.scatter(x_train[:, 1], y_train, color='blue', label='Training data')
    plt.scatter(x_eval[:, 1], y_pred, color='red', label='Validation predictions')
    plt.legend()
    
    # No need to save predictions
    # Plot data
    plt.show()
    # *** END CODE HERE ***


class LocallyWeightedLinearRegression():
    """Locally Weighted Regression (LWR).

    Example usage:
        > clf = LocallyWeightedLinearRegression(tau)
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def __init__(self, tau):
        super(LocallyWeightedLinearRegression, self).__init__()
        self.tau = tau
        self.x = None
        self.y = None

    def fit(self, x, y):
        """Fit LWR by saving the training set.

        """
        # *** START CODE HERE ***
        self.x = x
        self.y = y
        # *** END CODE HERE ***

    def predict(self, x):
        """Make predictions given inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        m = x.shape[0]
        y_pred = np.zeros(m)

        for i in range(m):
            diff = self.x - x[i]                    # (N, n)
            w = np.exp(-np.sum(diff[:, 1:] ** 2, axis=1) / (2 * self.tau ** 2))  
            # Weighted normal equation without forming full diagonal matrix
            XTWX = (self.x * w[:, None]).T @ self.x
            XTWy = (self.x * w[:, None]).T @ self.y
            theta = np.linalg.solve(XTWX + 1e-12 * np.eye(XTWX.shape[0]), XTWy)  
            y_pred[i] = x[i] @ theta
        return y_pred
        # *** END CODE HERE ***

if __name__ == '__main__':
    main(tau=5e-1,
         train_path='./train.csv',
         eval_path='./valid.csv')
