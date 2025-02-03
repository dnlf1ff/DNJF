import sys
import pickle

with open(f'{sys.argv[1]}_mlp.pkl','rb') as f:
    a = pickle.load(f)

print(a)
print(a.columns)
