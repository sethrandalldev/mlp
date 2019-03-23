import mlp

clsfyr = mlp.train(30, X_train, y_train)

results = mlp.classify(clsfyr, X_test)