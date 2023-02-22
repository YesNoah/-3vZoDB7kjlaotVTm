import os
print(os.getcwd())


from data.get_data import GetData
from data import make_dataset
from models.train_model import nnmodel
from models.generate_signals import buysell

result = nnmodel()
buysell(result)

