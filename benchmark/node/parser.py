import numpy as np
import pandas as pd

filename = 'results.txt'

f = open(filename, 'r')
lines = f.readlines()
f.close()

dataDict = dict()

for line in lines:
    elements = line.strip('\n\r').split(',')
    name  = elements[0]
    source = elements[2]
    target = elements[4]
    micro_f1 = eval(elements[6])
    macro_f1 = eval(elements[8])
    auc = eval(elements[10])
    if name not in dataDict:
        dataDict[name] = dict()
        if (source, target) not in dataDict[name]:
            dataDict[name][(source, target)] = [[micro_f1, macro_f1, auc]]
        else:
            dataDict[name][(source, target)].append([micro_f1, macro_f1, auc])
    else:
        if (source, target) not in dataDict[name]:
            dataDict[name][(source, target)] = [[micro_f1, macro_f1, auc]]
        else:
            dataDict[name][(source, target)].append([micro_f1, macro_f1, auc])
    
print('source target mean std:')

for k, v in dataDict.items():
    print(k)
    for st, value in v.items():
        value = np.array(value)
        mean_v = np.mean(value, axis=0)
        std_v = np.std(value, axis=0)
        print(st, mean_v, std_v)


# Create a pandas DataFrame from the nested dictionary
data = []

# Collect all (src, tgt) pairs
src_tgt_pairs = set()
for model_results in dataDict.values():
    src_tgt_pairs.update(model_results.keys())

# Sort the pairs for consistent ordering
src_tgt_pairs = sorted(src_tgt_pairs)

# Build the data for the DataFrame
for model, model_results in dataDict.items():
    row = {'Model': model}
    for pair in src_tgt_pairs:
        if pair in model_results:
            metrics = np.array(model_results[pair]) * 100  # Multiply by 100
            mean_metrics = np.mean(metrics, axis=0)
            std_metrics = np.std(metrics, axis=0)
            mean_metrics_rounded = np.round(mean_metrics, 2)
            std_metrics_rounded = np.round(std_metrics, 2)
            row[pair] = f"{mean_metrics_rounded} +/- {std_metrics_rounded}"
        else:
            row[pair] = "N/A"  # Handle cases where there are no results for this pair
    data.append(row)

# Create the DataFrame
df = pd.DataFrame(data)

# Set the 'Model' column as the index
df.set_index('Model', inplace=True)

# Optionally, save the DataFrame to a CSV file
df.to_csv('csv_results.csv')