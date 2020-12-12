import subprocess


def pre_train(lambda_coeff, epochs, lr, batch_size, aug, vocab_size):
    command = f'python train_full_imdb.py --lambda_coff={lambda_coeff} ' \
              f'--epochs={epochs} --lr={lr} --bsz={batch_size} --aug={aug} ' \
              f'--vocab={vocab_size}'
    return command


def small_regime(lambda_coeff, epochs, lr, batch_size, aug, vocab_size):
    command = f'python train_full_imdb.py --lambda_coff={lambda_coeff} ' \
              f'--epochs={epochs} --lr={lr} --bsz={batch_size} --aug={aug} ' \
              f'--vocab={vocab_size}'
    return command


def large_regime(lambda_coeff, epochs, lr, batch_size, aug, vocab_size):
    command = f'python train_full_imdb.py --lambda_coff={lambda_coeff} ' \
              f'--epochs={epochs} --lr={lr} --bsz={batch_size} --aug={aug} ' \
              f"--vocab={vocab_size} --prepath='imdb-pretrain'"
    return command


def run_command(bash_command):
    process = subprocess.Popen(bash_command.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()


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
