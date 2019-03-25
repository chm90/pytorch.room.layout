"""
Script for downloading the dataset files and pretrained network mentioned in the readme
"""

from os.path import join
from os import makedirs

from google_drive_downloader import GoogleDriveDownloader as gdd

DATA_DIR = 'datasets'

SUNRGBD_ID = '1oP0-n0AHW5mlfNrORLmQAAXqv0ByjIRg'
SUNRGBD_FILE = join(DATA_DIR, 'sunrgbd.zip')

LSUN_ID = '1e40AC_9CwgWPQL9eh18y2k9u4O0X3rl4'
LSUN_FILE = join(DATA_DIR, 'lsun.zip')

RL_PRETRAINED_ID = '1cyw3cfV4qPH2yS_XfeKDnJYbPERHA3tU'
RL_PRETRAINED_FILE = join('.', 'lsun.pth.tar')

ST_PRETRAINED_ID = '1-O45ENLICItubbah8osWkhe--BS-_of0'
ST_PRETRAINED_FILE = join('.', 'sunrgbd.pth.tar')

makedirs(DATA_DIR, exist_ok=True)

gdd.download_file_from_google_drive(
    file_id=SUNRGBD_ID, dest_path=SUNRGBD_FILE, unzip=True)
gdd.download_file_from_google_drive(
    file_id=LSUN_ID, dest_path=LSUN_FILE, unzip=True)
gdd.download_file_from_google_drive(
    file_id=ST_PRETRAINED_ID, dest_path=ST_PRETRAINED_FILE, unzip=False)
gdd.download_file_from_google_drive(
    file_id=RL_PRETRAINED_ID, dest_path=RL_PRETRAINED_FILE, unzip=False)
