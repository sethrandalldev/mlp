import mlp

X_train = [(0,1), (1,1)]
y_train = [1,0]
X_test = [(0,0), (1,0)]
y_test = [0, 1]

clsfyr = mlp.train(2, X_train, y_train)

results = mlp.classify(clsfyr, X_test)

print(results)