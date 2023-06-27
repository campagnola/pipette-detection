import os, threading, queue
import numpy as np
from PIL import Image


class TrainingData:
    def __init__(self, data_path=None, output_norm=None):
        self.data_path = data_path
        self.output_norm = output_norm

        if data_path is not None:
            self.load(data_path)

    def load(self, data_path):
        self.index = []
        fh = open(os.path.join(data_path, 'pos.csv'))
        for line in fh.readlines():
            img_file, z, row, col = line.split(',')
            self.index.append((img_file, float(z), float(row), float(col)))
        self.image_shape = self[0][0].shape

    def __len__(self):
        return len(self.index)
    
    def __getslice__(self, sl):
        images = []
        positions = []
        for i in range(*sl.indices(len(self))):
            img,pos = self[i]
            images.append(img)
            positions.append(pos)
        return np.stack(images), np.stack(positions)
        
    def __getitem__(self, item):
        if isinstance(item, slice):
            return self.__getslice__(item)
        img_file, z, row, col = self.index[item]
        img = np.asarray(Image.open(img_file)) / 255
        pos = np.array([z, row, col])
        if self.output_norm is not None:
            pos = self.output_norm(pos)
        return img, pos
    
    def generator(self, batch_size):
        preloader = Preloader(self, batch_size)
        try:
            while True:
                next_data = preloader.get_next()
                if next_data is None:
                    return
                img, pos = next_data
                yield img, pos / self.image_shape[0]
        finally:
            preloader.close()

    def split(self, proportions):
        start = 0
        parts = []
        for p in proportions:
            stop = start + int(len(self) * p)
            part = TrainingData(output_norm=self.output_norm)
            part.index = self.index[start:stop]
            part.image_shape = self.image_shape
            start = stop
            parts.append(part)
        return parts


class Normalizer:
    def __init__(self, possible_range):
        possible_range = np.array(possible_range)
        diff = possible_range[1] - possible_range[0]
        self.scale = 2 / diff
        self.offset = possible_range[0] + diff / 2

    def normalize(self, x):
        return (x - self.offset) * self.scale

    def denormalize(self, x):
        return (x / self.scale) + self.offset


class Preloader:
    def __init__(self, data, batch_size):
        self.data = data
        self.batch_size = batch_size
        self.queue = queue.Queue(maxsize=3)
        self.running = True
        self.thread = threading.Thread(target=self.preload, daemon=True)
        self.thread.start()

    def close(self):
        self.running = False
        while not self.queue.empty():
            self.queue.get()

    def preload(self):
        index = 0
        while self.running:
            stop = index + self.batch_size
            if stop > len(self.data):
                index = 0
                stop = index + self.batch_size
            chunk = self.data[index:stop]
            index = stop
            self.queue.put(chunk)
        self.queue.put(None)

    def get_next(self):
        if self.queue.empty():
            print("Warning: preloader queue is empty (this can slow down training)")
        return self.queue.get()
