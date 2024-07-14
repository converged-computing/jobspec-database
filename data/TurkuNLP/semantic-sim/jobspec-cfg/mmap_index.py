import pickle
import sys
import tqdm
import array
import mmap
import json
import random


class Qry:

    def __init__(self, filepref):
        with open(filepref+".meta", "rt") as f:
            self.meta = json.load(f)
        self.data = open(filepref+".data", "rb")
        self.data_mmap = mmap.mmap(
            self.data.fileno(), 0, access=mmap.ACCESS_READ)

        self.index = array.array("L")
        with open(filepref+".index", "rb") as f:
            self.index.fromfile(f, self.meta["len"])

        self.lengths = array.array("I")
        with open(filepref+".lengths", "rb") as f:
            self.lengths.fromfile(f, self.meta["len"])

    def get(self, i):
        idx = self.index[i]
        length = self.lengths[i]
        self.data_mmap.seek(idx)
        data = self.data_mmap.read(length)
        return pickle.loads(data)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--index-lines-to', metavar="FILEPREFIX",
                        help='Index lines from stdin, write to here. Will create FILEPREFIX.data and FILEPREFIX.index')
    parser.add_argument('--test-list', metavar="FILEPREFIX",
                        help='Just start listing random sentences, debugging stuff')
    args = parser.parse_args()

    if args.index_lines_to:
        with open(args.index_lines_to+".data", "wb") as f_data, open(args.index_lines_to+".index", "wb") as f_index, open(args.index_lines_to+".lengths", "wb") as f_lengths, open(args.index_lines_to+".meta", "wt") as f_meta:
            index = array.array("L")
            lengths = array.array("I")
            write_pos = 0
            for line in tqdm.tqdm(sys.stdin):
                line = line.rstrip("\n")
                data = pickle.dumps(line)
                index.append(write_pos)
                # remeber the byte position in the file and the length of the pickle
                lengths.append(len(data))
                write_pos += len(data)
                f_data.write(data)
            index.tofile(f_index)
            lengths.tofile(f_lengths)
            json.dump({"len": len(index)}, f_meta)

    elif args.test_list:
        qry = Qry(args.test_list)
        while True:
            i = random.randint(0, qry.meta["len"]-1)
            print(qry.get(i))
