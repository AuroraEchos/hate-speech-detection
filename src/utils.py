# This script provides utility functions for visualizing evaluation results.
# Date: 2024-10-12
# Name: Wenhao Liu

import json
import matplotlib.pyplot as plt
import numpy as np

def load_evaluation_report(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    return data

def plot_classification_report(report):
    classes = list(report.keys())[:-3]
    classes += ['macro avg', 'weighted avg'] 
    
    precision = [report[cls]['precision'] for cls in classes]
    recall = [report[cls]['recall'] for cls in classes]
    f1_score = [report[cls]['f1-score'] for cls in classes]
    support = [report[cls]['support'] for cls in classes]
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    index = np.arange(len(classes))
    bar_width = 0.2

    ax.bar(index, precision, bar_width, label='Precision')
    ax.bar(index + bar_width, recall, bar_width, label='Recall')
    ax.bar(index + 2 * bar_width, f1_score, bar_width, label='F1-Score')

    ax.set_xlabel('Class')
    ax.set_title('Classification Report Metrics')
    ax.set_xticks(index + bar_width)
    ax.set_xticklabels(classes, rotation=45, ha='right')
    ax.legend()

    plt.tight_layout()
    plt.show()


def plot_roc_curve(roc_data, auc):
    fpr = roc_data['false_positive_rate']
    tpr = roc_data['true_positive_rate']

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', label=f'ROC curve (AUC = {auc:.4f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.show()

def visualize_evaluation(filename):
    data = load_evaluation_report(filename)
    plot_classification_report(data['classification_report'])
    plot_roc_curve(data['roc_curve'], data['auc'])

visualize_evaluation('model\\evaluation_report.json')
