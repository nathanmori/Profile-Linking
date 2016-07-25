import psycopg2
from datetime import datetime
import pdb
from copy import deepcopy
import os


fname = '../data/tables.txt'
with open(fname, 'r') as fin:
    data = fin.read().splitlines(True)
data_new = deepcopy(data[1:])
data_new.append(data[0])
with open(fname, 'w') as fout:
    fout.writelines(data_new)

tablename = data[0].split('|')[1].strip()
print 'tablename: ', tablename



params = {
  'database': os.environ['TALENTFUL_PG_DATABASE'],
  'user': os.environ['TALENTFUL_PG_USER'],
  'password': os.environ['TALENTFUL_PG_PASSWORD'],
  'host': os.environ['TALENTFUL_PG_HOST'],
  'port': os.environ['TALENTFUL_PG_PORT']
}
conn = psycopg2.connect(**params)
c = conn.cursor()

#today = '2014-08-14'
#timestamp = datetime.strptime(today, '%Y-%m-%d').strftime("%Y%m%d")

query = '''
        SELECT *
        FROM ''' + tablename

c.execute(query)
a = c.fetchall()

for i in xrange(len(a)):
    print a[i]

print '# rows in', tablename, ':', len(a)

conn.commit()
conn.close()
