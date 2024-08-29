# update all files starting with mlp with suffix .pkl to .pt
import os

for file in os.listdir('.'):
    if file.startswith('mlp') and file.endswith('.pkl'):
        os.rename(file, file.replace('.pkl', '.pt'))
        print(file.replace('.pkl', '.pt'))