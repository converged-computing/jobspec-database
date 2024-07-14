import s2
import os
import multiprocessing as mp
import time
from random import randrange
import socket
from os.path import exists

sleep = randrange(1200)
#print(f'Sleep for: {sleep/60} minutes')
time.sleep(sleep)

MODE= 'dev'
USER = 'hpc-user'
PASSWORD = '-'
DEV = '-.us-east-1.rds.amazonaws.com'
PROD = '-'
LEVEL = 'L1C'

BUCKET = 'fire-prod'
KEY_ID = '-'
KEY = '-'
TOKEN = 'TOKEN'


ENDPOINT = DEV
SCHEMA = 'dev'
TABLE = 'prod_l1c_ca'

INSTANCE_ID = f'cpp-hpc-{socket.gethostname()}'
NUM_PROCESSES = 30
NUM_RECORDS = 180

os.environ["OMP_NUM_THREADS"] = "1" #fixes xarray issue

sen2cor_path = "/home/tjlogue/sen2cor/Sen2Cor-02.05.05-Linux64/bin/L2A_Process"
path_to_file = '/home/tjlogue/stop.txt'
file_exists = exists(path_to_file)

if __name__ ==  '__main__':
	while file_exists == False:
		try:
			sql_conn = s2.sql(mode= MODE, user = USER, password = PASSWORD, endpoint =ENDPOINT, schema = SCHEMA, table = TABLE, instance_id = INSTANCE_ID, num_processes= NUM_RECORDS, update = True)
			results = sql_conn.get_urls()
			if results == (): # () is returned when no data is selected
				break
			else:
				s2.mp_process_L1C(results, path= '/data03/home/tjlogue/L1C', bucket = BUCKET, sen2cor_path = sen2cor_path, sql_conn=sql_conn, num_cores=NUM_PROCESSES)
				file_exists = exists(path_to_file)
				break
		except:
			file_exists = exists(path_to_file)
			pass

	
