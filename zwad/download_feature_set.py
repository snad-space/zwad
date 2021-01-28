#!/usr/bin/env python3

import argparse
import hashlib
import json
import multiprocessing
import os
from itertools import repeat
from urllib.parse import urljoin

import requests


DEST_PATH = '.'
BASE_API_URL = 'https://zenodo.org/api/records/'
ZENODO_ID = 4318700


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-j', '--nproc', type=int, help='number of parallel processes')
    parser.add_argument('-o', '--output', default=DEST_PATH, help='destination path')
    parser.add_argument('-i', '--id', type=int, default=ZENODO_ID, help='Zenodo ID')
    args = parser.parse_args()
    return args


def download(description, destdir, session=requests):
    # zenodo_get maybe a better solution in general:
    #   https://gitlab.com/dvolgyes/zenodo_get/
    # but we'd like to practice a bit...

    url = description['links']['self']

    dest = os.path.join(destdir, description['key'])
    expected_size = description['size']

    # Download resuming is not possible at the moment.
    # Zenodo doesn't support it. So let's just test the size for equality.
    # See: https://github.com/zenodo/zenodo/issues/1599
    if not os.path.exists(dest) or expected_size != os.stat(dest).st_size:
        with session.get(url, stream=True) as response:
            print('Downloading {}'.format(dest))
            with open(dest, 'wb') as fh:
                for chunk in response:
                    fh.write(chunk)

    if not description['checksum'].startswith('md5:'):
        print('Skipping checksum of {} due to unknown hashing algorithm.'.format(dest))
        return

    expected_md5 = description['checksum'][4:]
    actual_md5 = md5sum(dest)
    if expected_md5 == actual_md5:
        print('Checksum test for {} PASSED.'.format(dest))
    else:
        raise RuntimeError('Checksum test for {} FAILED.'.format(dest))


def md5sum(filename):
    md5 = hashlib.md5()
    with open(filename, 'rb') as f:
        for block in iter(lambda: f.read(4096), b''):
            md5.update(block)

    return md5.hexdigest()


def execute_from_commandline():
    args = parse_args()

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    url = urljoin(BASE_API_URL, str(args.id))

    r = requests.get(url)
    if not r.ok:
        raise RuntimeError('Failed to download metadata from Zenodo: {}'.format(r.status_code))

    jsons = json.loads(r.text)
    files = jsons['files']

    with multiprocessing.Pool(processes=args.nproc) as pool:
        pool.starmap(download, zip(files, repeat(args.output)))


if __name__ == '__main__':
    execute_from_commandline()

