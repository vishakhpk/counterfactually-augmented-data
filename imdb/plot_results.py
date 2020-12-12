import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (roc_curve, roc_auc_score, accuracy_score, f1_score)

parser = argparse.ArgumentParser(description='Hello.')
parser.add_argument('--show', type=int, default=0)
args = parser.parse_args()

SHOW = args.show

small_names = [
    ('baseline factual',
     'epochs=20,lambda=0.0,lr=0.0005,vocab=3000,bsz=32,aug=0'),
    ('clp',
     'epochs=20,lambda=0.0005,lr=0.0005,vocab=3000,bsz=32,aug=1'),
    ('baseline augmented',
     'epochs=20,lambda=0.0,lr=0.0005,vocab=3000,bsz=32,aug=1'),
    ('clp augmented',
     'epochs=20,lambda=0.0007,lr=0.0005,vocab=3000,bsz=32,aug=1'),
]

large_names = [
    ('pretrain',
     'imdb-pretrain'),
    ('pretrain + baseline factual',
     'epochs=20,lambda=0.0,lr=0.0005,vocab=3000,bsz=32,aug=0+model-imdb-pretrain'),
    ('pretrain + baseline augmented',
     'epochs=20,lambda=0.0,lr=0.0005,vocab=3000,bsz=32,aug=1+model-imdb-pretrain'),
    ('pretrain + clp',
     'epochs=20,lambda=0.0005,lr=0.0005,vocab=3000,bsz=32,aug=0+model-imdb-pretrain'),
    ('pretrain + clp augmented',
     'epochs=20,lambda=0.0001,lr=0.0005,vocab=3000,bsz=32,aug=1+model-imdb-pretrain'),
]

for regime in [small_names, large_names]:
    # ROC curves
    plt.figure(figsize=(8, 8))
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    for (model_name, params) in regime:
        df = pd.read_csv(f'results/{params}.csv')

        y_true = df['y_true_fact']
        y_score = df['y_raw_fact']
        fpr_vals, tpr_vals, _ = roc_curve(y_true, y_score)
        auc_score = roc_auc_score(y_true, y_score)

        plt.plot(fpr_vals, tpr_vals, lw=2,
                 label=f'{model_name} (AUC = %0.2f)' % auc_score)

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.title(f'ROC Curves')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

    # Shrink current axis's height by 10% on the bottom
    ax = plt.gca()
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.2,
                        box.width, box.height * 0.8])

    # Put a legend below current axis
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.08),
              fancybox=True, shadow=True, ncol=1)

# plt.show()

plt.figure(figsize=(8, 6))
lambda_names = [
    # pretrain + clp augmented
    'epochs=30,lambda=1e-07,lr=0.0005,vocab=3000,bsz=32,aug=1+model-imdb-pretrain',
    'epochs=30,lambda=1e-06,lr=0.0005,vocab=3000,bsz=32,aug=1+model-imdb-pretrain',
    'epochs=30,lambda=1e-05,lr=0.0005,vocab=3000,bsz=32,aug=1+model-imdb-pretrain',
    'epochs=30,lambda=0.0001,lr=0.0005,vocab=3000,bsz=32,aug=1+model-imdb-pretrain',
    'epochs=30,lambda=0.001,lr=0.0005,vocab=3000,bsz=32,aug=1+model-imdb-pretrain',
    'epochs=30,lambda=0.01,lr=0.0005,vocab=3000,bsz=32,aug=1+model-imdb-pretrain',
    'epochs=30,lambda=0.0,lr=0.0005,vocab=3000,bsz=32,aug=1+model-imdb-pretrain',
]
lambdas = ['1e-07', '1e-06', '1e-05', '1e-04', '1e-03', '1e-02', '0']
for i, model_name in enumerate(lambda_names):
    df = pd.read_csv(f'results/f1-{model_name}.csv')

    f1_scores = df['f1_score'].rolling(window=5).median()
    steps = list(range(len(f1_scores)))
    plt.plot(steps, f1_scores, label=lambdas[i])
plt.legend()
plt.xlabel('Step')
plt.ylabel('F1 Score (Validation)')
plt.title('Learning Curve by Lambda -- Large Regime\nPretrain + CLP Augmented')
plt.ylim(0.2, 1)

# plt.show()

plt.figure(figsize=(8, 6))
lambda_names = [
    # clp augmented
    'epochs=30,lambda=1e-07,lr=0.0005,vocab=3000,bsz=32,aug=1',
    'epochs=30,lambda=1e-06,lr=0.0005,vocab=3000,bsz=32,aug=1',
    'epochs=30,lambda=1e-05,lr=0.0005,vocab=3000,bsz=32,aug=1',
    'epochs=30,lambda=0.0001,lr=0.0005,vocab=3000,bsz=32,aug=1',
    'epochs=30,lambda=0.001,lr=0.0005,vocab=3000,bsz=32,aug=1',
    'epochs=30,lambda=0.01,lr=0.0005,vocab=3000,bsz=32,aug=1',
    'epochs=30,lambda=0.0,lr=0.0005,vocab=3000,bsz=32,aug=1',
]
lambdas = ['1e-07', '1e-06', '1e-05', '1e-04', '1e-03', '1e-02', '0']
for i, model_name in enumerate(lambda_names):
    df = pd.read_csv(f'results/f1-{model_name}.csv')

    f1_scores = df['f1_score'].rolling(window=5).median()
    steps = list(range(len(f1_scores)))
    plt.plot(steps, f1_scores, label=lambdas[i])
plt.legend()
plt.xlabel('Step')
plt.ylabel('F1 Score (Validation)')
plt.title('Learning Curve by Lambda -- Small Regime\nCLP Augmented')
plt.ylim(0.2, 1)


for regime in [small_names, large_names]:
    for (model_name, params) in regime:
        df = pd.read_csv(f'results/{params}.csv')
        y_true_fact = df['y_true_fact']
        y_score_fact = df['y_raw_fact']
        y_score_cfact = df['y_raw_cfact']
        y_pred_fact = df['y_pred_fact']
        y_pred_cfact = df['y_pred_cfact']
        print('-'*80)
        print(model_name)
        print(params)
        print(f'AUC: {roc_auc_score(y_true_fact, y_pred_fact)}')
        print(f'Accuracy: {accuracy_score(y_true_fact, y_pred_fact)}')
        print(f'F1 Score: {f1_score(y_true_fact, y_pred_fact)}')
        print(f'CF Consistency: {np.not_equal(y_pred_fact, y_pred_cfact).mean()}')
        mean_difference = np.abs(np.subtract(y_score_fact, y_score_cfact)).mean()
        print(f'CF Gap: {mean_difference}')
        print('-'*80)


if SHOW:
    plt.show()
