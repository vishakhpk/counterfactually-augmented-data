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
    lambda_coeff = 0
    epochs = 20
    lr = 0.0005
    batch_size = 32
    aug = 0
    vocab_size = 3000

    command = large_regime(lambda_coeff, epochs, lr, batch_size, aug, vocab_size)

    run_command(command)


if __name__ == '__main__':
    main()
