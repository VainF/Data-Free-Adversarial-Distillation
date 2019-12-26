import argparse
import csv
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--log', type=str)

args = parser.parse_args()

best_acc_list = []
acc_array = []
max_acc =0.0
with open(args.log, 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        row = list(map(float, row))
        best_acc = np.max(row)
        best_acc_list.append(best_acc)
        acc_array.append(row)
        if best_acc > max_acc:
            max_acc = best_acc
    print(best_acc_list)
    print("Mean=%.4f, Std=%.4f, Best=%.4f"%(np.mean(best_acc_list), np.std( best_acc_list), max_acc))