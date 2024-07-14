print("importing")
import nltk
import parsl

import os
import csv
import json
import logging
import sys
import traceback

import parsl_configs

stdout = sys.stdout
stderr = sys.stderr

sys.stdout = sys.stderr = open('stdout.txt', 'a')

subjectsDir = 'data'
cleanedFile = 'processed.json'

maxRunning = 256
perBatch = 500

csv.field_size_limit(sys.maxsize)

def display(inputS, end = '\n'):
    inputS = str(inputS)
    stdout.write(inputS)
    stdout.write(end)
    stdout.flush()

def resetStdout():
    sys.stdout = stdout
    sys.stderr = stderr

def subjectIter():
    for e in os.scandir(subjectsDir):
        if e.name.endswith('.tsv'):
            with open(e.path) as f:
                reader = csv.DictReader(f, delimiter = '\t', quoting = csv.QUOTE_NONE)
                for row in reader:
                    try:
                        yield {
                            'wos_id' : row['wos_id'],
                            'title' : row['title'],
                            'abstract' : row['abstract'],
                            }
                    except:
                        pass
    return

def gen_full_tokenizer(dfk):
    #Start the parsl generator instead of using the wrapper syntax
    return parsl.App('python', dfk)(tokenizeEntry)


def tokenizeEntry(entries):
    import nltk
    import json

    def tokenizer(target):
        try:
            return nltk.word_tokenize(str(target).lower())
        except:
            return []
            #import pdb; pdb.set_trace()

    def sentinizer(sent):
        try:
            return [tokenizer(s) for s in nltk.sent_tokenize(sent)]
        except TypeError:
            #Missing abstract
            return []

    ret = []
    for e in entries:
        ret.append({
                'wos_id' : e['wos_id'],
                'title' : tokenizer(e['title']),
                'abstract' : sentinizer(e['abstract']),
                })
    return '\n'.join([json.dumps(r) for r in ret])

def checkRunning(running):
    completesNameLists = [k for k, v in running.items() if v.done()]
    succCount = 0
    if len(completesNameLists) > 0:
        with open(cleanedFile, 'a') as f:
            for name in completesNameLists:
                try:
                    results = running.pop(name).result()
                except Exception as e:
                    display("{}\t{}".format(e, traceback.format_exc()))
                else:
                    f.write(results)
                    f.write('\n')
                    succCount += 50
    return succCount

def main():
    display("Loading DFK")
    parsl.set_file_logger("parsl.log", level=logging.DEBUG)

    dfk = parsl.DataFlowKernel(config=parsl_configs.rccNodeExclusive)

    display("Loading App")
    full_app = gen_full_tokenizer(dfk)

    display("Loading data iter")
    datIter = subjectIter()

    display("Starting run")

    running = {}
    done = False
    succCount = 0
    doneIter = False
    try:
        while not done:
            #Only add maxRunning jobs to the queue at once
            while len(running) < maxRunning:
                batch = []
                for i in range(perBatch):
                    try:
                        batch.append(next(datIter))
                    except StopIteration:
                        #If none left just skip adding
                        doneIter = True
                        break
                batchName = batch[0]['wos_id']
                running[batchName] = full_app(batch)
            succCount += checkRunning(running)
            display("Completed {}".format(succCount))
            #End the loop if all jobs are done and no more can be added
            if doneIter and len(running) < 1:
                done = True

    except KeyboardInterrupt:
        display("Closing down")
        dfk.cleanup()
        raise
    except:
        resetStdout()
        raise

    dfk.cleanup()
    display("Done")

if __name__ == '__main__':
    main()
