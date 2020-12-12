import subprocess


def pre_train(lambda_coeff, epochs, lr, batch_size, aug, vocab_size):
    command = f'python train_full_imdb.py --epochs={epochs} ' \
              f' --lr={lr} --batch_size={batch_size} --vocab_size={vocab_size}'
    return command


def small_regime(lambda_coeff, epochs, lr, batch_size, aug, vocab_size):
    command = f'python run_imdb.py --lambda_coeff={lambda_coeff} ' \
              f'--epochs={epochs} --lr={lr} --batch_size={batch_size} ' \
              f'--aug={aug} --vocab_size={vocab_size}'
    return command


def large_regime(lambda_coeff, epochs, lr, batch_size, aug, vocab_size):
    command = f'python run_imdb.py --lambda_coeff={lambda_coeff} ' \
              f'--epochs={epochs} --lr={lr} --batch_size={batch_size} ' \
              f'--aug={aug} --vocab_size={vocab_size} ' \
              f'--prepath=model-imdb-pretrain.pt'
    return command


def run_command(bash_command):
    # process = subprocess.Popen(bash_command.split(), stdout=subprocess.PIPE)
    # output, error = process.communicate()
    subprocess.run(bash_command)


def main():
    small_param_sets = {
        'baseline_factual':  [
            {'lambda_coeff': 0, 'aug': 0,  'epochs': 20, 'lr': 0.0005,
             'batch_size': 32, 'vocab_size': 3000}
        ],
        'baseline_augmented':  [
            {'lambda_coeff': 0, 'aug': 1,  'epochs': 20, 'lr': 0.0005,
             'batch_size': 32, 'vocab_size': 3000}
        ],
        'clp': [
            {'lambda_coeff': 0.0005, 'aug': 0,  'epochs': 20, 'lr': 0.0005,
             'batch_size': 32, 'vocab_size': 3000}
        ],
        'clp_augmented':  [
            {'lambda_coeff': 0.000001, 'aug': 1,  'epochs': 30, 'lr': 0.0005,
             'batch_size': 32, 'vocab_size': 3000},
            {'lambda_coeff': 0.00001, 'aug': 1,  'epochs': 30, 'lr': 0.0005,
             'batch_size': 32, 'vocab_size': 3000},
            {'lambda_coeff': 0.0001, 'aug': 1,  'epochs': 30, 'lr': 0.0005,
             'batch_size': 32, 'vocab_size': 3000},
            {'lambda_coeff': 0.001, 'aug': 1,  'epochs': 30, 'lr': 0.0005,
             'batch_size': 32, 'vocab_size': 3000},
            {'lambda_coeff': 0.01, 'aug': 1,  'epochs': 30, 'lr': 0.0005,
             'batch_size': 32, 'vocab_size': 3000},
        ],
    }
    pretrain_param_sets = {
        'pretrain':  [
            {'epochs': 20, 'lr': 0.0005, 'batch_size': 32, 'vocab_size': 3000}
        ],
    }
    large_param_sets = {
            'pretrain_baseline_factual':  [
                {'lambda_coeff': 0, 'aug': 0,  'epochs': 20, 'lr': 0.0005,
                 'batch_size': 32, 'vocab_size': 3000}
            ],
            'pretrain_baseline_augmented':  [
                {'lambda_coeff': 0, 'aug': 1,  'epochs': 20, 'lr': 0.0005,
                 'batch_size': 32, 'vocab_size': 3000}
            ],
            'pretrain_clp': [
                {'lambda_coeff': 0.0005, 'aug': 0,  'epochs': 20,
                 'lr': 0.0005, 'batch_size': 32, 'vocab_size': 3000}
            ],
            'pretrain_clp_augmented':  [
                {'lambda_coeff': 0.0001, 'aug': 1,  'epochs': 20,
                 'lr': 0.0005, 'batch_size': 32, 'vocab_size': 3000}
            ],
        }

    for params in small_param_sets['clp_augmented']:
        command = small_regime(**params)
        # command = large_regime(**params)
        run_command(command)


if __name__ == '__main__':
    main()
