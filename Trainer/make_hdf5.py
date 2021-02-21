import sys
import glob
import multiprocessing as mp
import h5py
import numpy as np


def process_one(file):
    with open(file, 'rt') as f:
        lines = f.readlines()

    width, height = map(int, lines[0].split())
    result = int(lines[1])

    features = []
    probs = []
    values = []

    for line_id in range(2, len(lines), 2):
        feat_str = lines[line_id]
        prob_str = lines[line_id + 1]

        feature = np.fromstring(
            feat_str.strip(), dtype=np.int8, sep=' ').reshape(16, height, width)
        prob = np.fromstring(prob_str.strip(), sep=' ')

        features.append(feature)
        probs.append(prob)
        values.append([result])

    features = np.array(features)
    probs = np.array(probs)
    values = np.array(values)

    return features, probs, values


def task(self_file_list, save_file, lock):
    for file in self_file_list:
        features, probs, values = process_one(file)

        lock.acquire()

        with h5py.File(save_file, 'a') as root:
            m = root['feature'].shape[0]
            n = len(features)

            root['feature'].resize(m + n, axis=0)
            root['feature'][m:m+n] = features

            root['prob'].resize(m + n, axis=0)
            root['prob'][m:m+n] = probs

            root['val'].resize(m + n, axis=0)
            root['val'][m:m+n] = values

        lock.release()


if __name__ == '__main__':
    if len(sys.argv) < 4:
        print('usage: python3 {} <dir> [thread num]'.format(sys.argv[0]))
        sys.exit()

    thread_num = 1
    if len(sys.argv) == 5:
        thread_num = int(sys.argv[4])

    dir_name = sys.argv[1]
    self_file_list = glob.glob('{}/*'.format(dir_name))
    save_file = dir_name+'.hdf5'

    width = int(sys.argv[2])
    height = int(sys.argv[3])

    with h5py.File(save_file, 'w') as root:
        root.create_dataset('feature', shape=(0, 16, height, width), maxshape=(None, 16, height, width),
                            dtype=np.int8, compression='lzf')
        root.create_dataset('prob', shape=(0, 5), maxshape=(
            None, 5), dtype=np.float32, compression='lzf')
        root.create_dataset('val', shape=(0, 1), maxshape=(
            None, 1), dtype=np.int8, compression='lzf')

    lock = mp.Lock()

    if thread_num == 1:
        task(self_file_list, save_file, lock)
    else:
        procs = []
        length = int(np.ceil(len(self_file_list) / thread_num))

        for rank in range(thread_num):
            file_list = self_file_list[rank * length:(rank+1) * length]
            p = mp.Process(target=task, args=(file_list, save_file, lock))
            procs.append(p)
            p.start()

        for p in procs:
            p.join()

    print('{} has been created.'.format(save_file))
    print()
