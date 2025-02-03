import sys
import pickle

with open(f'{sys.argv[1]}_mlp.pkl','rb') as f:
    a = pickle.load(f)

print(a)
print(a.columns)
a['system'] = [sys.argv[1]]*3

with open(f'{sys.argv[1]}_mlp.pkl','wb')as f:
    pickle.dump(a,f)
