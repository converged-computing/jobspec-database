from Naked.toolshed.shell import execute_js, muterun_js
import sqlite3
import sys
import json
import time, datetime

def main():

    conn = sqlite3.connect('sevendwarfs-11-1.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE data
                 (date text, wait real, status text, active text)''')

    #run for a week
    f = open('data-11-1.txt','a')
    
    while True:
        #sleep until 7 AM
	'''
	t = datetime.datetime.today()
	future = datetime.datetime(t.year, t.month, t.day, 7, 0)
	if t.hour >= 7:
	    future += datetime.timedelta(days=1)
	time.sleep((future-t).total_seconds())
	'''
        #do 7am stuff
    
	waits = []
	times = []
	status = []
	active = []
	for i in range(68):

	    response = muterun_js('sevenDwarfs.js')
	    if response.exitcode == 0:

		try:
		    parsed_json = json.loads(response.stdout)
	            #print(parsed_json[11])
	            #print(parsed_json[11]['waitTime'])

		    '''
		    waits.append(parsed_json[11]['waitTime'])
		    times.append(datetime.datetime.today())
		    status.append(parsed_json[11]['status'])
		    active.append(parsed_json[11]['active'])
		    '''

		    c.execute("INSERT INTO data VALUES (?, ?, ?, ?)",
			      (str(datetime.datetime.today()), parsed_json[11]['waitTime'], parsed_json[11]['status'], str(parsed_json[11]['active'])))
		    conn.commit()
			  

		    f.write(str(datetime.datetime.today()) +'\t')
		    f.write(str(parsed_json[11]['waitTime']) + '\t')
		    f.write(parsed_json[11]['status'] + '\t')
		    f.write(str(parsed_json[11]['active']) +'\n')
		    f.flush()
		except ValueError:
		    pass

	    else:
		sys.stderr.write(response.stderr)

	    time.sleep(900)

    f.close()
    conn.close()
    #print waits
    #print times
    #print status
    #print active

if __name__ == '__main__':
    main()
