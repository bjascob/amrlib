#!/usr/bin/python3
import os
import setup_run_dir    # run script 2 levels up
from   amrlib import download
from   amrlib import defaults

if __name__ == '__main__':
    server_url = 'http://127.0.0.1:8000'

    print('Be sure the file server is ready for %s' % server_url)
    print('Run this locally with python3 -m http.server 8000')
    if os.path.isdir(defaults.data_dir):
        print('NOTE !! %s exists.  Are you sure you want to continue?' % defaults.data_dir)
    input('Press enter to continue> ')
    print()

    download('model_stog',  os.path.join(server_url, 'model_parse_gsii-v0_0_0.tar.gz'))
    print()
    download('model_gtos', os.path.join(server_url, 'model_generate_t5-v0_0_0.tar.gz'))
    print()
