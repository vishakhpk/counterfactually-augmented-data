import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

models_to_plot = [
    # baseline factual
    'epochs=20,lambda=0.0,lr=0.0005,vocab=3000,bsz=32,aug=0',
    # baseline augmented
    'epochs=20,lambda=0.0,lr=0.0005,vocab=3000,bsz=32,aug=1',
    # clp
    'epochs=20,lambda=0.0005,lr=0.0005,vocab=3000,bsz=32,aug=1',
    # clp augmented
    'epochs=20,lambda=0.0007,lr=0.0005,vocab=3000,bsz=32,aug=1',
    # pretrain
    'imdb-pretrain',
    # pretrain + baseline factual
    'epochs=20,lambda=0.0,lr=0.0005,vocab=3000,bsz=32,aug=0+model-imdb-pretrain',
    # pretrain + baseline augmented
    'epochs=20,lambda=0.0,lr=0.0005,vocab=3000,bsz=32,aug=1+model-imdb-pretrain',
    # pretrain + clp
    'epochs=20,lambda=0.0005,lr=0.0005,vocab=3000,bsz=32,aug=0+model-imdb-pretrain',
    # pretrain + clp augmented
    'epochs=20,lambda=0.0001,lr=0.0005,vocab=3000,bsz=32,aug=0+model-imdb-pretrain',
]

# ROC curves
plt.figure(figsize=(8, 6))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
for model_name in models_to_plot:
    df = pd.read_csv(f'results/{model_name}.csv')

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
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
          fancybox=True, shadow=True, ncol=5)

plt.show()
