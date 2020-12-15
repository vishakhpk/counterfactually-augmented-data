import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (roc_curve, roc_auc_score, accuracy_score, f1_score)

parser = argparse.ArgumentParser(description='Hello.')
parser.add_argument('--show', type=int, default=0)
parser.add_argument('--save', type=int, default=0)
parser.add_argument('--window', type=int, default=5)
args = parser.parse_args()

SHOW = args.show
SAVE = args.save
WINDOW = args.window

fig_num = 0

filename_strings = {  # camel case to support latex output
    'baselineFactual':  [
        # {'lambda_coeff': 0, 'aug': 0,  'epochs': 30, 'lr': 0.0005,
        #  'batch_size': 32, 'vocab_size': 3000, 'aug_test': aug_test}
        f'epochs=30,lambda=0.0,lr=0.0005,vocab=3000,bsz=32,aug=0,aug_test=1'
    ],
    'NEWTESTbaselineFactual':  [
        # {'lambda_coeff': 0, 'aug': 0,  'epochs': 30, 'lr': 0.0005,
        #  'batch_size': 32, 'vocab_size': 3000, 'aug_test': aug_test}
        f'epochs=30,lambda=0.0,lr=0.0005,vocab=3000,bsz=32,aug=0,aug_test=0'
    ],
    'baselineAugmented':  [
        # {'lambda_coeff': 0, 'aug': 1,  'epochs': 30, 'lr': 0.0005,
        #  'batch_size': 32, 'vocab_size': 3000, 'aug_test': aug_test}
        f'epochs=30,lambda=0.0,lr=0.0005,vocab=3000,bsz=32,aug=1,aug_test=1'
    ],
    'NEWTESTbaselineAugmented':  [
        # {'lambda_coeff': 0, 'aug': 1,  'epochs': 30, 'lr': 0.0005,
        #  'batch_size': 32, 'vocab_size': 3000, 'aug_test': aug_test}
        f'epochs=30,lambda=0.0,lr=0.0005,vocab=3000,bsz=32,aug=1,aug_test=0'
    ],
    'clp': [
        # {'lambda_coeff': lambda_coeff, 'aug': 0,  'epochs': 30, 'lr': 0.0005,
        #  'batch_size': 32, 'vocab_size': 3000, 'aug_test': aug_test}
        f'epochs=30,lambda={lambda_coeff},lr=0.0005,vocab=3000,bsz=32,aug=0,aug_test=1'
         for lambda_coeff in ['1e-07', '1e-06', '1e-05', '0.0001', '0.001', '0.01', '0.0']
    ],
    'NEWTESTclp': [
        # {'lambda_coeff': lambda_coeff, 'aug': 0,  'epochs': 30, 'lr': 0.0005,
        #  'batch_size': 32, 'vocab_size': 3000, 'aug_test': aug_test}
        f'epochs=30,lambda={lambda_coeff},lr=0.0005,vocab=3000,bsz=32,aug=0,aug_test=0'
         for lambda_coeff in ['1e-07', '1e-06', '1e-05', '0.0001', '0.001', '0.01', '0.0']
    ],
    'clpAugmented':  [
        # {'lambda_coeff': lambda_coeff, 'aug': 1,  'epochs': 30, 'lr': 0.0005,
        #  'batch_size': 32, 'vocab_size': 3000, 'aug_test': aug_test}
        f'epochs=30,lambda={lambda_coeff},lr=0.0005,vocab=3000,bsz=32,aug=1,aug_test=1'
        for lambda_coeff in ['1e-07', '1e-06', '1e-05', '0.0001', '0.001', '0.01', '0.0']
    ],
    'NEWTESTclpAugmented':  [
        # {'lambda_coeff': lambda_coeff, 'aug': 1,  'epochs': 30, 'lr': 0.0005,
        #  'batch_size': 32, 'vocab_size': 3000, 'aug_test': aug_test}
        f'epochs=30,lambda={lambda_coeff},lr=0.0005,vocab=3000,bsz=32,aug=1,aug_test=0'
        for lambda_coeff in ['1e-07', '1e-06', '1e-05', '0.0001', '0.001', '0.01', '0.0']
    ],
}

small_regime = [(key, name) for key, filenames in filename_strings.items()
                for name in filenames]
large_regime = [(key + 'Pretrain', name + '+model-imdb-pretrain')
                for key, filenames in filename_strings.items()
                for name in filenames] # + [('pretrain', 'imdb-pretrain')]

regimes = [small_regime, large_regime]

best_results = {key: (0, None) for key in filename_strings.keys()}
best_results.update({key + 'Pretrain': (0, None) for key in filename_strings.keys()})
best_results.update({'pretrain': 'imdb-pretrain'})

latex_output = {key: [] for key in best_results.keys()}

best_results = {
    'aug_test=0': best_results.copy(),
    'aug_test=1': best_results.copy(),
}

latex_output = {
    'aug_test=0': latex_output.copy(),
    'aug_test=1': latex_output.copy(),
}
for regime in regimes:
    for (model_name, param_string) in regime:
        df = pd.read_csv(f'results/{param_string}.csv')
        y_true_fact = df['y_true_fact']
        y_score_fact = df['y_raw_fact']
        y_score_cfact = df['y_raw_cfact']
        y_pred_fact = df['y_pred_fact']
        y_pred_cfact = df['y_pred_cfact']
        print('-'*80)
        print(model_name)
        print(param_string)
        auc = roc_auc_score(y_true_fact, y_score_fact)
        accuracy = accuracy_score(y_true_fact, y_pred_fact)
        f1 = f1_score(y_true_fact, y_pred_fact)
        cf_consistency = np.not_equal(y_pred_fact, y_pred_cfact).mean()
        cf_gap = np.abs(np.subtract(y_score_fact, y_score_cfact)).mean()
        print(f'AUC: {auc}')
        print(f'Accuracy: {accuracy}')
        print(f'F1 Score: {f1}')
        print(f'CF Consistency: {cf_consistency}')
        print(f'CF Gap: {cf_gap}')
        print('-'*80)

        if 'NEWTEST' in model_name:
            which_results = 'aug_test=0'
        else:
            which_results = 'aug_test=1'

        if auc > best_results[which_results][model_name][0]:
            best_results[which_results][model_name] = (auc, param_string)

            results = {
                'Auc': auc,
                'Accy': accuracy,
                'F': f1,
                'Consist': cf_consistency,
                'Gap': cf_gap,
            }
            latex_output[which_results][model_name] = [f'% {model_name}\n% {param_string}']
            for name, value in results.items():
                latex_string = '\\newcommand{\\' + f'{model_name}{name}' '}{' + f'{value:.4f}' + '}'
                latex_output[which_results][model_name].append(latex_string)

for which_results, test_results in latex_output.items():
    print(f'% ---------- {which_results} ---------- %')
    for model, latex_strings in test_results.items():
        for latex_string in latex_strings:
            print(latex_string)
    print(f'% ------------------------------------ %')

for regime in regimes:
    # ROC curves
    plt.figure(figsize=(8, 8))
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    for (model_name, param_string) in regime:
        df = pd.read_csv(f'results/{param_string}.csv')

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

plt.figure(figsize=(4, 6))
lambda_names_aug = [name + '+model-imdb-pretrain'
                    for name in filename_strings['clpAugmented']]
lambda_names = [name + '+model-imdb-pretrain'
                for name in filename_strings['clp']]

lambdas = ['1e-07', '1e-06', '1e-05', '1e-04', '1e-03', '1e-02', '0']
for i, model_name in enumerate(lambda_names_aug):
    df = pd.read_csv(f'results/f1-{model_name}.csv')

    f1_scores = df['f1_score'].rolling(window=WINDOW).median()
    steps = [step/2 for step in list(range(len(f1_scores)))]
    plt.plot(steps[40:], f1_scores[40:], label=lambdas[i])
plt.legend()
# plt.xlabel('Epoch')
# plt.yticks([])
# plt.ylabel('F1 Score (Validation)')
# plt.title('Learning Curve by Lambda -- Large Regime\nPretrain + CLP Augmented')
plt.ylim(0.75, 0.9)

if SAVE:
    fig_num += 1
    plt.savefig(f'fig_{fig_num}_pretrain_clp_aug_zoom.pdf')

plt.figure(figsize=(8, 6))
for i, model_name in enumerate(lambda_names):
    df = pd.read_csv(f'results/f1-{model_name}.csv')

    f1_scores = df['f1_score'].rolling(window=WINDOW).median()
    steps = [step/2 for step in list(range(len(f1_scores)))]
    plt.plot(steps, f1_scores, label=lambdas[i])
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('F1 Score (Validation)')
# plt.title('Learning Curve by Lambda -- Large Regime\nPretrain + CLP')
plt.ylim(0.5, 1)

if SAVE:
    fig_num += 1
    plt.savefig(f'fig_{fig_num}_pretrain_clp.pdf')

# plt.show()

plt.figure(figsize=(8, 6))
lambda_names_aug = filename_strings['clpAugmented']
lambda_names = filename_strings['clp']
lambdas = ['1e-07', '1e-06', '1e-05', '1e-04', '1e-03', '1e-02', '0']
for i, model_name in enumerate(lambda_names_aug):
    df = pd.read_csv(f'results/f1-{model_name}.csv')

    f1_scores = df['f1_score'].rolling(window=WINDOW).median()
    steps = [step/2 for step in list(range(len(f1_scores)))]
    plt.plot(steps, f1_scores, label=lambdas[i])
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('F1 Score (Validation)')
# plt.title('Learning Curve by Lambda -- Small Regime\nCLP Augmented')
plt.ylim(0.2, 1)

if SAVE:
    fig_num += 1
    plt.savefig(f'fig_{fig_num}_clp_aug.pdf')

plt.figure(figsize=(8, 6))
for i, model_name in enumerate(lambda_names):
    df = pd.read_csv(f'results/f1-{model_name}.csv')

    f1_scores = df['f1_score'].rolling(window=WINDOW).median()
    steps = [step/2 for step in list(range(len(f1_scores)))]
    plt.plot(steps, f1_scores, label=lambdas[i])
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('F1 Score (Validation)')
# plt.title('Learning Curve by Lambda -- Small Regime\nCLP')
plt.ylim(0.2, 1)

if SHOW:
    plt.show()

if SAVE:
    fig_num += 1
    plt.savefig(f'fig_{fig_num}_clp.pdf')
