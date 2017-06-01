def nnet(train, target, valid):

    ds = SupervisedDataSet(52-len(NU)+5, 1)

    for i in range(len(train)):
        ds.appendLinked(train[i], target[i])

    n = RecurrentNetwork()

    n.addInputModule(LinearLayer(52-len(NU)+5, name='in'))
    n.addModule(SigmoidLayer(3, name='hidden'))
    n.addOutputModule(LinearLayer(1, name='out'))

    n.addConnection(FullConnection(n['in'], n['hidden'], name='c1'))
    n.addConnection(FullConnection(n['hidden'], n['out'], name='c2'))
    n.addRecurrentConnection(FullConnection(n['hidden'], n['hidden'], name='c3'))

    n.sortModules()

    t = BackpropTrainer(n,learningrate=0.001,verbose=True)
    t.trainOnDataset(ds, 20)

    prediction = np.zeros((11573, 1), dtype = int)
    for i in range(11573):
        prediction[i] = n.activate(valid[i])

    return prediction