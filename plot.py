"""plot.py — plot ERM vs mixup test-accuracy curves from the run logs
and print the comparison against the paper's numbers.

Expects:
    runs/erm_s0/log.csv
    runs/mixup_s0/log.csv
Produces compare.png and a console summary.
"""
import csv
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def read_log(path):
    epochs, accs, train_losses = [], [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            epochs.append(int(row['epoch']))
            accs.append(float(row['test_acc']) * 100)
            train_losses.append(float(row['train_loss']))
    return epochs, accs, train_losses


def main():
    runs = [('ERM (baseline)', 'runs/erm_s0'), ('mixup (alpha=1.0)', 'runs/mixup_s0')]
    results = {}
    plt.figure(figsize=(8, 5))
    for name, d in runs:
        log = os.path.join(d, 'log.csv')
        if not os.path.exists(log):
            print(f'skip {name}: {log} not found')
            continue
        ep, acc, trl = read_log(log)
        results[name] = (max(acc), acc[-1], trl[-1])
        plt.plot(ep, acc, label=name)
    plt.xlabel('Epoch')
    plt.ylabel('Test accuracy (%)')
    plt.title('ERM vs mixup on CIFAR-10 (PreAct ResNet-18, 100 epochs)')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('compare.png', dpi=150)
    print('saved compare.png')
    for name, (best, final, trl) in results.items():
        print(f'{name}: best test_acc = {best:.2f}%  final = {final:.2f}%  final train_loss = {trl:.4f}')
    if len(results) == 2:
        gap = results['mixup (alpha=1.0)'][0] - results['ERM (baseline)'][0]
        print(f'gap (best): mixup is {gap:+.2f} points above ERM   [paper, 200 epochs: +1.40]')


if __name__ == '__main__':
    main()
