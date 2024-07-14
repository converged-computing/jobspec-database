import subprocess
import sys

map_file = sys.argv[1]
index = int(sys.argv[2])

f = open(map_file, 'r')
for i in range(index-1):
 f.readline()
arr = f.readline().split(';')
CODE_FILE = arr[1]
ROOT_FOLDER = arr[2]
FILE_NAME = arr[3]
OUTPUT_FILE = arr[4]
f.close()


p = subprocess.Popen(['python3', CODE_FILE, ROOT_FOLDER, FILE_NAME, OUTPUT_FILE], stdout=subprocess.PIPE)
output, error = p.communicate()
print(output)

