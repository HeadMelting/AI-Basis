import os

def search(dirname):
    filenames = os.listdir(dirname)
    for filename in filenames:
        full_filename = os.path.join(dirname,filename)
        ext = os.path.splitext(full_filename)[-1]
        if os.path.isdir(full_filename):
            search(full_filename)
        if ext == ".py":
            print(full_filename)

search('/Users/hyunsukim/Documents/GitHub/python-sutdy/')