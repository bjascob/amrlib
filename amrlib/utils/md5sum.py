#!/usr/bin/python3
import sys
import hashlib


# Run-time is about the same as the system md5sum and may 
# be just fractionally faster
# Results with 851M binary file..
# python3 md5sum.py     1.20 seconds
# system  md5sum.py     1.23 seconds
# use first_chunk_only to only use the first chunksize worth of data
# to create the md5sum
def md5sum(fn, chunksize=2**20, first_chunk_only=False):
    file_hash = hashlib.md5()
    with open(fn, 'rb') as f:
        while True:
            chunk = f.read(chunksize)
            if not chunk: break
            file_hash.update(chunk)
            if first_chunk_only: break
    return file_hash.hexdigest()


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: md5sum.py <filename>')
        exit()
    fn = sys.argv[1]

    if 0:
        print('md5sum on full file')
        print( md5sum(fn) )
    else:
        print('md5sum on first chunk only')
        print( md5sum(fn, first_chunk_only=True) )
