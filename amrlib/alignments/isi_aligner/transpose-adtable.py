#!/usr/bin/python3
import sys
import numpy


def transpose_adtable(ain_fn, din_fn, aout_fn, dout_fn):
    TSIZE = 220
    HALF  = 110
    a = numpy.zeros(shape=(TSIZE, TSIZE, TSIZE), dtype='double')
    d = numpy.zeros(shape=(TSIZE, TSIZE, TSIZE), dtype='double')

    # eg.. 1 1 3 100 0.25
    with open(ain_fn) as fin:
        for line in fin:
            parts = line.strip().split()
            a1, a2, a3, a4 = [int(a) for a in parts[:4]]
            a[a1][a2][a3] = float(parts[4])

    # eg.. 1 1 3 100 0.25
    with open(din_fn) as fin:
        for line in fin:
            parts = line.strip().split()
            a1, a2, a3, a4 = [int(a) for a in parts[:4]]
            d[a1][a2][a4] = float(parts[4])

    # for (int i = 0; i < 110; i++) for (int j = 0; j < 110; j++) for (int k = 0; k < 110; k++) if (d[k][j][i] > 0) {
    #   fout0 << j << " " << k << " " << i << " " << "100" << " " << d[k][j][i] << endl;
    with open(aout_fn, 'w') as fout:
        for i in range(HALF):
            for j in range(HALF):
                for k in range(HALF):
                    if d[k][j][i] > 0:
                        fout.write('%d %d %d %d %f\n' % (j, k, i, 100, d[k][j][i]))

    # for (int i = 0; i < 110; i++) for (int j = 0; j < 110; j++) for (int k = 0; k < 110; k++) if (a[k][j][i] > 0) {
    #   fout1 << j << " " << k << " "  << "100" << " " << i << " " << a[k][j][i] << endl;
    with open(dout_fn, 'w') as fout:
        for i in range(HALF):
            for j in range(HALF):
                for k in range(HALF):
                    if a[k][j][i] > 0:
                        fout.write('%d %d %d %d %f\n' % (j, k, 100, i, a[k][j][i]))


# From cpp code transpose-adtable.cpp
if __name__ == '__main__':
    ain_fn, din_fn, aout_fn, dout_fn = sys.argv[1:5]
    transpose_adtable(ain_fn, din_fn, aout_fn, dout_fn)
