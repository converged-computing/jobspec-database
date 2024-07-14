import os, sys
import numpy as np
from scipy.io import wavfile

try:
  import cPickle as pickle
except:
  print 'cPickle not available'
  import pickle

SOURCE_DIR = 'genres_processed'

def readGenre(path, genre):
  path = os.path.join(path, genre)
  fileNames = os.listdir(path)
  ret = {}

  print 'Reading: {0}'.format(genre)
  stdout = sys.stdout
  stdout.write('[' + '-' * 100 + ']')
  progress = 0

  for f in fileNames:
    progress += 1
    start, end = f.find('.'), f.find('_')
    idx = int(f[start+1:end])
    start, end = f.find('clip'), f.find('.wav')
    clipNum = int(f[start+4:end])

    p = int(float(progress) / len(fileNames) * 100)
    s = ('#' * p).ljust(100, '-')
    stdout.write('\r[{0}] {1}%'.format(s, p))
    stdout.flush()

    _, data = wavfile.read(os.path.join(path, f))
    try:
      ret[idx].append((clipNum, data))
    except:
      ret[idx] = [(clipNum, data)]
  stdout.write('\n')

  def extractArray(tup):
    return tup[1]

  for k in ret.keys():
    arrays = sorted(ret[k], key=lambda x: x[0])
    ret[k] = map(extractArray, arrays)

  return ret

if __name__ == '__main__':
  currPath = os.getcwd()
  sourcePath = os.path.join(currPath, SOURCE_DIR)
  genres = os.listdir(sourcePath)

  data = {}
  for g in genres:
    data[g] = readGenre(sourcePath, g)

  print 'Writing into file: data.in'
  o = open('data.in', 'w')
  pickle.dump(data, o)
  o.close()
