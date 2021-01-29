#!/usr/bin/python3
import sys
import numpy

def transpose_ttable(t0_fn, t1_fn, tout_fn):
    TSIZE=11000
    p = numpy.zeros(shape=(TSIZE, TSIZE), dtype='double')

    with open(t0_fn) as fin0:
        for line in fin0:
            a, b, val = line.strip().split()
            if int(a) == 0:
                p[0][int(b)] = float(val)
            else:
                break

    with open(t1_fn) as fin1:
        for line in fin1:
            a, b, val = line.strip().split()
            if int(b) != 0:
                p[int(b)][int(a)] = float(val)

    i, j = numpy.nonzero(p)
    max_i, max_j = max(i), max(j)
    with open(tout_fn, 'w') as fout:
        for i in range(max_i + 1):
            for j in range(max_j + 1):
                if p[i][j] > 0.000000001:
                    fout.write('%d %d %f\n' % (i, j, p[i][j]))


# From cpp code transpose-ttable.cpp
if __name__ == '__main__':
    t0_fn, t1_fn, tout_fn = sys.argv[1:4]
    transpose_ttable(t0_fn, t1_fn, tout_fn)
