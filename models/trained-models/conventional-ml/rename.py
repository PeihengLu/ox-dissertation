# rename all files in the current dir starting with ridge_regression to ridge
import os

for filename in os.listdir('.'):
    if filename.startswith('ridge_regression'):
        os.rename(filename, filename.replace('ridge_regression', 'ridge'))
        print(filename.replace('ridge_regression', 'ridge'))