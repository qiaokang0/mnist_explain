import torch
import os
import numpy as np
import codecs


def get_int(b):
    return int(codecs.encode(b, 'hex'), 16)

def read_image_file(path):
    '''
    comment like this would follow the indent rule
    # mnist file is a binary file , and the format explanation is on the official web.(with a simple header and hex for the gray shade)
    This func transform the binary file to a variable!
    keyword:
    f.read()
    int(codecs.encode(b, 'hex'), 16)
    numpy.frombuffer
    torch.from_numpy
    view
    '''
    with open(path, 'rb') as f:
        data = f.read()
        assert get_int(data[:4]) == 2051
        length = get_int(data[4:8])
        num_rows = get_int(data[8:12])
        num_cols = get_int(data[12:16])
        parsed = np.frombuffer(data, dtype=np.uint8, offset=16)
        return torch.from_numpy(parsed).view(length, num_rows, num_cols)


def read_label_file(path):
    with open(path, 'rb') as f:
        data = f.read()
        assert get_int(data[:4]) == 2049
        length = get_int(data[4:8])
        parsed = np.frombuffer(data, dtype=np.uint8, offset=8)
        return torch.from_numpy(parsed).view(length).long()

pro=r'/Users/joe/Documents/raw'

training_set = (
    read_image_file(os.path.join(pro, 'train-images-idx3-ubyte')),
    read_label_file(os.path.join(pro, 'train-labels-idx1-ubyte'))
)
test_set = (
            read_image_file(os.path.join(pro, 't10k-images-idx3-ubyte')),
            read_label_file(os.path.join(pro, 't10k-labels-idx1-ubyte'))
)

with open(os.path.join(pro, 'train.pt'), 'wb') as f:
    torch.save(training_set, f)
    '''only torch.load can retrieve the variable, cause the return type are the same 
    '''
with open(os.path.join(pro, 'test.pt'), 'wb') as f:
    torch.save(test_set, f)

data, targets = torch.load(os.path.join(pro, 'train.pt'))
print(len(data))
print(len(targets))
print('end1')
print(type(training_set[1]))
print('end2')
