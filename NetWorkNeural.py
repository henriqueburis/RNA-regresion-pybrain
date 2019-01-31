from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer


ds = SupervisedDataSet(2,1) # duas entradas u,a saida

ds.addSample((0.8,0.4),(0.7)) # amostra de entradas e saida
ds.addSample((0.5,0.7),(0.5))
ds.addSample((1.0,0.8),(0.95))


nn = buildNetwork(2, 4, 1, bias=True)

trainer = BackpropTrainer(nn, ds)

for i in range(0, 2000):
    print(trainer.train())

#while True:
    #dormi = float('Domiu')
    #estudo = float('EStudou')

z = nn.activate((0.8,0.5))
print(str(z))
