import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,confusion_matrix, classification_report
from load_data import load_dataset
import joblib



class SVMClassifier:
    """
        A comprehensive class to handle the entire Support Vector Machine (SVM)
        workflow for handwritten digit classification, including data loading,
        training (with optional GridSearchCV), evaluation, visualization, and persistence.
    """


    def __init__(self, use_gridsearch=True):
        # Initializes the SVMClassifier object and sets up placeholders for data and results.
        # use_gridsearch(bool): Whether to perform GridSearchCV for hyperparameter tuning.
        self.use_gridsearch = use_gridsearch
        self.model = None             # Stores the final trained SVC model or the best estimator from GridSearchCV
        self.best_params_ = None      # Stores the optimal hyperparameters found by GridSearchCV

        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.y_pred_train = None
        self.y_pred_test = None
        self.cm_test = None         # Confusion Matrix for the test set



    # -----------
    # Load data
    # -----------
    def load_data(self, train_size, test_size, img_size):
        X_train, y_train, X_test, y_test = load_dataset(train_size, test_size, img_size)
        self.X_train = np.array(X_train)
        self.y_train = np.array(y_train)
        self.X_test = np.array(X_test)
        self.y_test = np.array(y_test)



    # ------------------------------------------
    # Train model (with optional GridSearchCV)
    # ------------------------------------------
    def train(self):

        if self.use_gridsearch:
            # Hyperparameter Tuning with GridSearch
            params = {
                'kernel': ['rbf'],
                'C': [1, 5, 7, 10],
                'gamma': ['scale', 'auto', 0.01, 0.001]
            }

            # Initialize K-Fold Cross-Validation
            kf = KFold(n_splits=5, shuffle=True, random_state=42)

            grid = GridSearchCV(
                SVC(),
                param_grid=params,
                cv=kf,
                scoring='accuracy',
                verbose=1
            )

            grid.fit(self.X_train, self.y_train)
            self.model = grid.best_estimator_
            self.best_params_ = grid.best_params_
            print(f'Best params from GridSearchCV: {self.best_params_}')
            print(f'Best CV score: {grid.best_score_}')

        else:
            # Training with Fixed Parameters
            self.model = SVC(C=7, gamma='scale', kernel='rbf')
            self.model.fit(self.X_train, self.y_train)

        # Predictions
        self.y_pred_train = self.model.predict(self.X_train)
        self.y_pred_test = self.model.predict(self.X_test)



    # -----------------------------------------------------------------------------------
    # Calculates key performance metrics for a given set of true labels and predictions.
    # -----------------------------------------------------------------------------------
    def compute_metrics(self, y_true, y_pred):
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='macro')
        recall = recall_score(y_true, y_pred, average='macro')
        f1 = f1_score(y_true, y_pred, average='macro')
        clr = classification_report(y_true, y_pred)
        return accuracy, precision, recall, f1, clr



    # ------------------------------------
    # Print training and testing metrics
    # ------------------------------------
    def print_metrics(self):
        # Compute metrics for both sets
        acc_train, prec_train, rec_train, f1_train, clr_train = self.compute_metrics(
            self.y_train, self.y_pred_train
        )
        acc_test, prec_test, rec_test, f1_test, clr_test = self.compute_metrics(
            self.y_test, self.y_pred_test
        )

        print('- - - - - - -  Train Metrics - - - - - - -')
        print(f'Train Metrics:\n'
              f'Accuracy: {acc_train}\n'
              f'Precision: {prec_train}\n'
              f'Recall: {rec_train}\n'
              f'F1-Score: {f1_train}\n'
              f'Classification Report:\n{clr_train}')

        print('\n- - - - - - - Test Metrics - - - - - - -')
        print(f'Test Metrics:\n'
              f'Accuracy: {acc_test}\n'
              f'Precision: {prec_test}\n'
              f'Recall: {rec_test}\n'
              f'F1-Score: {f1_test}\n'
              f'Classification Report:\n{clr_test}')

        return (acc_train, prec_train, rec_train, f1_train), (acc_test, prec_test, rec_test, f1_test)



    # ------------------------------------------------
    # Plot the confusion matrix for test predictions
    # ------------------------------------------------
    def plot_confusion_matrix(self):
        self.cm_test = confusion_matrix(self.y_test, self.y_pred_test)
        plt.figure(figsize=(8,6))
        sns.heatmap(self.cm_test, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.show()



    # --------------------------------------------------------
    # Plot a bar chart comparing training and testing metrics
    # --------------------------------------------------------
    def plot_metrics(self):
        acc_train, prec_train, rec_train, f1_train, _ = self.compute_metrics(self.y_train, self.y_pred_train)
        acc_test, prec_test, rec_test, f1_test, _ = self.compute_metrics(self.y_test, self.y_pred_test)

        metrics = ['Accuracy', 'Precision', 'Recall', 'F1']
        train_scores = [acc_train, prec_train, rec_train, f1_train]
        test_scores = [acc_test, prec_test, rec_test, f1_test]

        x = range(len(metrics))
        plt.figure(figsize=(8,5))
        plt.bar(x, train_scores, width=0.4, label='Train', align='center')
        plt.bar(x, test_scores, width=0.4, label='Test', align='edge')
        plt.xticks(x, metrics)
        plt.ylim(0,1.05)
        plt.ylabel('Score')
        plt.title('Train vs Test Metrics')
        plt.legend()
        plt.show()



    # -------------------------------------------------------------------
    # Plot a bar chart showing the number of wrong predictions per class
    # -------------------------------------------------------------------
    def plot_wrong_predictions(self):
        if self.cm_test is None:
            self.cm_test = confusion_matrix(self.y_test, self.y_pred_test)
        errors = np.sum(self.cm_test, axis=1) - np.diag(self.cm_test)
        classes = [str(i) for i in range(10)]

        plt.figure(figsize=(8,5))
        plt.bar(classes, errors, color='r')
        plt.xlabel('Class')
        plt.ylabel('Number of wrong predictions')
        plt.title('Wrong predictions per class')
        plt.show()



    # ----------------------------------------------
    # Save the trained model to a file using joblib
    # ----------------------------------------------
    def save_model(self, filename='svm_digit_model.joblib'):
        joblib.dump(self.model, filename)
        print(f'Model saved to {filename}')




# --------------
# Example usage
# --------------
if __name__ == '__main__':
    clf = SVMClassifier(use_gridsearch=False)
    clf.load_data(50000, 10000, 25)
    clf.train()
    clf.print_metrics()
    clf.plot_confusion_matrix()
    clf.plot_metrics()
    clf.plot_wrong_predictions()
    clf.save_model()