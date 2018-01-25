import numpy as np
import os
from PIL import Image
from random import shuffle
import json

DATASET_FOLDER = "./Datasets/TumorDataset/"

POSITIVE_ROOT = DATASET_FOLDER + "Positive/"
NEGATIVE_ROOT = DATASET_FOLDER + "Negative/"

TRAINING_JSON = DATASET_FOLDER + "train_data.json"
TEST_JSON = DATASET_FOLDER + "test_data.json"

data = []
for (dirpath, dirnames, filenames) in os.walk(NEGATIVE_ROOT):
    for filename in filenames:
        file_dict = {
            'file_name': NEGATIVE_ROOT + "//" + filename,
            'label': 0
        }
        data.append(file_dict)

for (dirpath, dirnames, filenames) in os.walk(POSITIVE_ROOT):
    for filename in filenames:
        file_dict = {
            'file_name': POSITIVE_ROOT + "//" + filename,
            'label': 1
        }
        data.append(file_dict)

shuffle(data)
training_data = data[:int(0.8 * len(data))]
test_data = data[int(0.8 * len(data)):]

f = open(TRAINING_JSON, "w")
f.write(json.dumps(training_data, indent=4, sort_keys=True))
f.close()

f = open(TEST_JSON, 'w')
f.write(json.dumps(test_data, indent=4, sort_keys=True))
f.close()


def training_batch_generator(batch_size):

    for batch_index in range(int(len(training_data) / batch_size)):
        image_list = []
        labels = []

        for training_row in training_data[batch_index * batch_size:(batch_index + 1) * batch_size]:
            im = Image.open(training_row['file_name'])
            im = im.convert('L')
            width, height = im.size

            image_np = np.array(im.getdata(), dtype=np.float32)
            image_np = image_np.reshape(width, height, 1)

            image_list.append(image_np)

            labels.append([training_row['label']])

        yield batch_index, np.array(image_list), np.array(labels)


def get_training_length():
    return len(training_data)


def test_batch_generator(batch_size):

    for batch_index in range(len(test_data) // batch_size):

        image_list = []
        labels = []

        for test_row in test_data[batch_index * batch_size: (batch_index + 1) * batch_size]:
            im = Image.open(test_row['file_name'])
            im = im.convert('L')

            width, height = im.size

            image_np = np.array(im.getdata(), dtype=np.float32)
            image_np = image_np.reshape(width, height, 1)

            image_list.append(image_np)

            labels.append([test_row['label']])

        yield batch_index, np.array(image_list), np.array(labels)


def main():
    pass

if __name__ == '__main__':
    main()
