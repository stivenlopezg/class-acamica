import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, \
    precision_score, recall_score, f1_score, roc_auc_score


class ModelEvaluation:
    """
    Evaluation a classification model with main metrics evaluation
    """

    def __init__(self, observed: list or pd.Series, predicted: list or pd.Series):
        self.observed = observed
        self.predicted = predicted
        self.metrics = None

    def generate_report(self):
        """
        Generate a DataFrame with the main classification metrics
        :return: pd.DataFrame
        """
        report = np.round(pd.DataFrame(classification_report(y_true=self.observed,
                                                             y_pred=self.predicted, output_dict=True)), 2).T
        return report

    def confusion_matrix(self, normalize: bool = True):
        """
        Generates a confusion matrix
        :return: pd.DataFrame
        """
        if normalize:
            table = np.round(pd.crosstab(index=self.observed, columns=self.predicted,
                                         rownames=['Observed'], colnames=['Predicted'], normalize='index'), 2)
        else:
            table = np.round(pd.crosstab(index=self.observed, columns=self.predicted,
                                         rownames=['Observed'], colnames=['Predicted']), 2)
        return table

    def calculate_metrics(self):
        """
        Calculate the main classification metrics
        :return: dict
        """
        metrics = {
            'roc_auc': np.round(roc_auc_score(y_true=self.observed, y_score=self.predicted), 2),
            'accuracy': np.round(accuracy_score(y_true=self.observed, y_pred=self.predicted), 2),
            'precision': np.round(precision_score(y_true=self.observed, y_pred=self.predicted, average='weighted'), 2),
            'recall': np.round(recall_score(y_true=self.observed, y_pred=self.predicted, average='weighted'), 2),
            'f1': np.round(f1_score(y_true=self.observed, y_pred=self.predicted, average='weighted'), 2),
        }
        self.metrics = metrics
        return self.metrics

    def print_metrics(self):
        """
        Print a text summary of the main classification metrics
        :return: string
        """
        print(f'El AUC es: {self.metrics["roc_auc"]}')
        print(f'El accuracy es: {self.metrics["accuracy"]}')
        print(f'La precision es: {self.metrics["precision"]}')
        print(f'El recall es: {self.metrics["recall"]}')
        print(f'El F1 Score es: {self.metrics["f1"]} \n')
        return None
