from os import path, listdir, remove
from tqdm import tqdm
import numpy as np
from datetime import date
from multiprocessing import Process, SimpleQueue, Event, Value
from save.database import Database
from arborescence.arborescence import Folders


class Writer(Process):

    def __init__(self, db_folder, db_name, queue, shutdown):

        Process.__init__(self)

        self.db = Database(folder=db_folder, name=db_name)
        self.queue = queue
        self.shutdown = shutdown
        self.counter = Value('i', 0)

    def get_tables_names(self):

        return self.db.get_tables_names()

    def run(self):

        already_existing_tables = self.db.get_tables_names()

        while not self.shutdown.is_set():

            try:

                param = self.queue.get()

                if param is not None:
                    if param[0] == "write":

                        table_name, columns, content = param[1]

                        if table_name in already_existing_tables:
                            self.db.remove_table(table_name=table_name)
                            self.db.connexion.commit()

                        self.db.create_table_and_write_n_rows(table_name=table_name, columns=columns,
                                                              array_like=content)

                        already_existing_tables.append(table_name)

                    elif param[0] == "remove_db":

                        old_db_name = param[1]
                        remove(old_db_name)
                        self.counter.value += 1
                    else:
                        raise Exception("Bad argument for writer queue.")
                else:
                    break

            except KeyboardInterrupt:

                print()
                print("Writer grab keyboard interrupt.")
                if not self.db.is_close:
                    self.db.close()

        if not self.db.is_close:
            self.db.close()
        self.shutdown.set()
        print("Writer: DEAD.")


class Reader(Process):

    def __init__(self, db_folder, db_to_merge, queue, shutdown):

        Process.__init__(self)
        self.db_folder = db_folder
        self.db_to_merge = db_to_merge
        self.shutdown = shutdown
        self.writer_queue = queue

    def run(self):

        for db_name in self.db_to_merge:

            if not self.shutdown.is_set():

                db = Database(folder=self.db_folder, name=db_name)

                try:

                    db_tables_names = db.get_tables_names()

                    for table_name in db_tables_names:

                        if not self.shutdown.is_set():
                            columns = db.get_columns(table_name)
                            content = db.read_n_rows(table_name=table_name, columns=columns)
                            self.writer_queue.put(["write", [table_name, columns, content]])
                        else:
                            break

                    db.close()

                    old_db_name = "{}/{}.db".format(self.db_folder, db_name)
                    self.writer_queue.put(["remove_db", old_db_name])

                except KeyboardInterrupt:
                    print()
                    print("Reader grab keyboard interrupt.")
                    break

            else:
                break

        if not self.shutdown.is_set():
            self.writer_queue.put(None)

        print("Reader: DEAD.")


def merge_db(db_folder, new_db_name, db_to_merge):

    assert path.exists(db_folder), '`{}` is a wrong path to db folder, please correct it.'.format(db_folder)

    shutdown = Event()
    writer_queue = SimpleQueue()

    writer = Writer(db_folder=db_folder, db_name=new_db_name, queue=writer_queue, shutdown=shutdown)
    reader = Reader(db_folder=db_folder, db_to_merge=db_to_merge,
                    queue=writer_queue, shutdown=shutdown)

    reader.start()
    writer.start()

    pbar = tqdm(total=len(db_to_merge))

    c = 0
    while not shutdown.is_set():
        try:
            new_c = writer.counter.value
            progress = new_c - c
            if progress > 0:
                pbar.update(progress)
                c = new_c
            Event().wait(2)

        except KeyboardInterrupt:
            print()
            print("Main thread grab the keyboard interrupt")
            break

    shutdown.set()
    pbar.close()
    # writer.join()
    # reader.join()

    print("writer alive", writer.is_alive())
    print("reader alive", reader.is_alive())

    if writer.is_alive():

        print("Waiting writer...")
        writer.join()

    print("WRITER EXECUTED")

    if reader.is_alive():
        print("Waiting reader...")
        writer_queue.get()
        print("Waiting reader 2...")
        reader.join()

    print("READER EXECUTED")

    print("Done.")


def merge_all_db_from_same_folder(db_folder, new_db_name):

    # Be sure that the path of the folder containing the databases is correct.
    assert path.exists(db_folder), 'Wrong path to db folder, please correct it.'

    # Get the list of all the databases
    list_db_name = [i[:-3] for i in listdir(db_folder) if i[-3:] == ".db" and i[:-3] != new_db_name]
    assert len(list_db_name), 'Could not find any db...'

    merge_db(db_folder=db_folder, new_db_name=new_db_name, db_to_merge=list_db_name)


def example_of_merging_db_from_list_of_db():

    new_db = "new_db"
    data_folder = "../data"
    db_to_merge = ["2016_09_29_idx17", "2016_09_29_idx16"]
    merge_db(db_folder=data_folder, new_db_name=new_db, db_to_merge=db_to_merge)


def main():

    db_folder = Folders.folders["data"]
    # new_db_name = "data_{}".format(str(date.today()).replace("-", "_"))
    new_db_name = "data_2016_10_12"

    merge_all_db_from_same_folder(db_folder=db_folder, new_db_name=new_db_name)


if __name__ == "__main__":

    main()