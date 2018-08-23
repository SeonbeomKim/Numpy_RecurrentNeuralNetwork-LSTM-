import numpy as np
import csv

def save_csv(path, data):
	with open(path, 'w', newline='') as o:
		wr = csv.writer(o)
		for line in data:
			wr.writerow(line)


# [a:0, b:1, ... z:25]
alphabet_len = 26
alphabet = {chr(ord('a')+i):i for i in range(alphabet_len)}
#keys = np.array(list(alphabet.keys()))
keys = list(alphabet.keys())
print(alphabet)
pick = 4

dataset = []
for i in range(130000): #중복데이터 때문에 조금 더 뽑음.
	sample = np.random.choice(keys, pick, replace=False) # pick개 추출하는데 중복없이 뽑아라. [h, k, u, j]
	values = np.array([alphabet[k] for k in sample]) # [7 11 21 10]
	target = np.argsort(values) # [0 3 1 2] 의미는 7 10 11 21 순이라는 것.

	data = np.concatenate((values, target))
	dataset.append(data.tolist())

print(len(dataset))

#dup remove
dataset = np.unique(dataset, axis=0)[:100000]

#shuffle
np.random.shuffle(dataset)

train = dataset[:70000]
vali = dataset[70000:85000]
test = dataset[85000:]



print(len(train), len(vali), len(test))

save_csv('./train.csv', train)
save_csv('./vali.csv', vali)
save_csv('./test.csv', test)