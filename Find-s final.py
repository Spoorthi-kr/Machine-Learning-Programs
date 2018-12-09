import csv
with open('Training_examples.csv') as csvFile:
    data = [line[:-1] for line in csv.reader(csvFile) if line[-1] == "Yes"]
print("POSITIVE EXAMPLES ARE:{}",*(n for n in data),sep="\n")
S = ['ɸ']*len(data[0])
print("output in each steps are:\n{}".format(S))
for example in data:
    i = 0
    for feature in example:
        S[i] = feature if S[i] == 'ɸ' or S[i] == feature else '?'
        i += 1
    print(S)
print("\nThe maximally specific Find-s hypothesis for the given training examples is\n",S)







