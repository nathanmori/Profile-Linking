#! /bin/bash

# try conda install first, pip

sudo apt-get install awscli
sudo apt-get install ipython
sudo apt-get install git
sudo apt-get install yum
sudo apt-get install gcc
sudo apt-get install python-setuptools
sudo apt-get install python-pip
sudo apt-get install postgres-xc
sudo apt-get install libpq-dev python-dev
pip install psycopg2
sudo pip install pandas
sudo apt-get ipython-notebook
apt-get install libsm6 libxrender1 libfontconfig1
conda install seaborn
sudo apt-get build-dep matplotlib
pip -U matplotlib

#>>> import matplotlib
#>>> matplotlib.matplotlib_fname()
# This is the file location in Ubuntu
#'/etc/matplotlibrc'
#Then modify the backend in that file to Agg. That is it.

#this might negate the need for some of the above lines
wget http://repo.continuum.io/archive/Anaconda2-4.1.1-Linux-x86_64.sh
bash Anaconda2-4.1.1-Linux-x86_64.sh


git clone https://github.com/nathanmori/Linking-Profiles-Across-Social-Media-Sites-By-User.git
