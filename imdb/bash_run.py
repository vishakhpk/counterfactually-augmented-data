import subprocess


def pre_train(lambda_coeff, epochs, lr, batch_size, aug, vocab_size, aug_test):
    command = f'python train_full_imdb.py --epochs={epochs} ' \
              f' --lr={lr} --batch_size={batch_size} --vocab_size={vocab_size} ' \
              f'--aug_test={aug_test}'
    return command


def small_regime(lambda_coeff, epochs, lr, batch_size, aug, vocab_size, aug_test):
    command = f'python run_imdb.py --lambda_coeff={lambda_coeff} ' \
              f'--epochs={epochs} --lr={lr} --batch_size={batch_size} ' \
              f'--aug={aug} --vocab_size={vocab_size} --aug_test={aug_test}'
    return command


def large_regime(lambda_coeff, epochs, lr, batch_size, aug, vocab_size, aug_test):
    command = f'python run_imdb.py --lambda_coeff={lambda_coeff} ' \
              f'--epochs={epochs} --lr={lr} --batch_size={batch_size} ' \
              f'--aug={aug} --vocab_size={vocab_size} --aug_test={aug_test} ' \
              f'--prepath=model-imdb-pretrain.pt'
    return command


def run_command(bash_command):
    # process = subprocess.Popen(bash_command.split(), stdout=subprocess.PIPE)
    # output, error = process.communicate()
    subprocess.run(bash_command)


def main():
    param_sets = {
        'baseline_factual':  [
            {'lambda_coeff': 0, 'aug': 0,  'epochs': 30, 'lr': 0.0005,
             'batch_size': 32, 'vocab_size': 3000, 'aug_test': aug_test}
            for aug_test in [0, 1]
        ],
        'baseline_augmented':  [
            {'lambda_coeff': 0, 'aug': 1,  'epochs': 30, 'lr': 0.0005,
             'batch_size': 32, 'vocab_size': 3000, 'aug_test': aug_test}
            for aug_test in [0, 1]
        ],
        'clp': [
            {'lambda_coeff': lambda_coeff, 'aug': 0,  'epochs': 30, 'lr': 0.0005,
             'batch_size': 32, 'vocab_size': 3000, 'aug_test': aug_test}
             for lambda_coeff in [1e-07, 1e-06, 1e-05, 1e-04, 1e-03, 1e-02, 0]
             for aug_test in [0, 1]
        ],
        'clp_augmented':  [
            {'lambda_coeff': lambda_coeff, 'aug': 1,  'epochs': 30, 'lr': 0.0005,
             'batch_size': 32, 'vocab_size': 3000, 'aug_test': aug_test}
            for lambda_coeff in [1e-07, 1e-06, 1e-05, 1e-04, 1e-03, 1e-02, 0]
            for aug_test in [0, 1]
        ],
    }
    pretrain_param_sets = [
            {'epochs': 30, 'lr': 0.0005, 'batch_size': 32, 'vocab_size': 3000,
             'aug_test': aug_test}
            for aug_test in [0, 1]
    ]

    # run pre-training first
    for params in pretrain_param_sets:
        command = pre_train(**params)
        run_command(command)

    # sweep same parameters in small and large regime
    for experiment in param_sets.keys():
        for params in experiment:
            command = small_regime(**params)
            run_command(command)

            command = large_regime(**params)
            run_command(command)


if __name__ == '__main__':
    main()
