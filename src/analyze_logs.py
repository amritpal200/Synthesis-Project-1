# -------------- to run: 
# install wazuh: https://documentation.wazuh.com/current/quickstart.html
# run: systemctl start wazuh-manager
# open wazuh manager: https://localhost:55000
# enter user "admin" and password
# find all credentials by running on path where installed: tar -O -xvf wazuh-install-files.tar wazuh-install-files/wazuh-passwords.txt
# run on this repo root: sudo /var/ossec/framework/python/bin/python3 src/analyze_logs.py

import numpy as np
import os
from scripts.wazuh_logtest import WazuhLogtest

n = 8
#TODO set log_file and out_file as parameters
log_file = f'/media/eric/D/cloud_eric/universitat/4t_semestre/synthesis_project/logs/sitges_access.2024012{n}.log'
out_file = f'/media/eric/D/repos/Synthesis-Project-1/data/sitges_access.2024012{n}_level.csv'

wazuh_logtest = WazuhLogtest()

session_token = None
with open(log_file, 'r') as file:
	logs = file.readlines()

total_lines = len(logs)
levels = np.zeros(total_lines, dtype=int)

for i, log in enumerate(logs):
	if i % 10000 == 0:
		print(f"i = {i}")
	try:
		response = wazuh_logtest.process_log(log, token=session_token)
		level = response['output']['rule']['level'] if 'rule' in response['output'] else None
		session_token = response['token']
		if level is None:
			levels[i] = -1
		else:
			levels[i] = level
	except Exception as e:
		print(f"Error processing log: {e}")
		levels[i] = -1
		print(f"Log: {log}")
		break

# create output file if it does not exist
if not os.path.exists(out_file):
	with open(out_file, 'w') as file:
		file.write("")
np.savetxt(out_file, levels, delimiter=',', fmt='%d')

# print statistics
histogram = np.histogram(levels, bins=np.arange(-1, 17))
for i, count in enumerate(histogram[0]):
	print(f"Level {histogram[1][i]}: {count} logs")
print(f"Total logs: {len(levels)}")
print(f"Total logs with no level: {np.sum(levels == -1)}")
