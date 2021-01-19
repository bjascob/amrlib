#!/usr/bin/python3
import io
import shutil
import numpy


class GetAlignments(object):
    def __init__(self):
        self.tokens = []    # strings
        self.level  = []    # ints
        self.tree   = numpy.zeros(shape=(1000,100), dtype='int')
        self.deg    = numpy.zeros(shape=(1000,),    dtype='int')
        self.p      = numpy.zeros(shape=(1000,),    dtype='int')
        self.fout   = None

    @classmethod
    def from_amr_aligned(cls, infn):
        self = cls()
        with open(infn) as fin:
            lines = [l.strip() for l in fin]
        lines = [l for l in lines if l and not l.startswith('#')]
        self.build_alignment_strings(lines)
        return self

    @classmethod
    def from_amr_strings(cls, strings):
        self = cls()
        self.build_alignment_strings(strings)
        return self

    def write_to_file(self, outfn):
        with open(outfn, 'w') as f:
            self.fout.seek(0)
            shutil.copyfileobj(self.fout, f)

    def get_alignments(self):
        return self.fout.getvalue().splitlines()


    ###########################################################################
    #### Private methods for building the alignments from a file
    ###########################################################################

    def build_alignment_strings(self, lines):
        self.fout = io.StringIO()
        for line in lines:
            self.parse(line)
            self.print_tree(0, '1')
            self.fout.write('\n')

    def print_tree(self, r, l):
        s = ''
        tkn = self.tokens[r]
        for i in range(len(tkn)):
            if tkn[i] == '~':
                stkn = tkn[i+3:]
                al = ''
                for j in range(len(stkn)):
                    if stkn[j] == ',':
                        if tkn[0] == ':':
                            self.fout.write('%s-%s.r ' % (al, l))
                        else:
                            self.fout.write('%s-%s ' % (al, l))
                        al = ''
                    else:
                        al = al + stkn[j]
                if tkn[0] == ':':
                    self.fout.write('%s-%s.r ' % (al, l))
                else:
                    self.fout.write('%s-%s ' % (al, l))
                break
        if self.tokens[r] == '/':
            s = s + '/' + self.print_tree(self.tree[r][0], l)
            return s
        if self.tokens[r] == ':':
            s = s + self.tokens[r] + ' ' + self.print_tree(self.tree[r][0], l)
            return s
        if self.deg[r] > 0:
            s = s + '('
        s = s + self.tokens[r] + ' '
        for i in range(self.deg[r]):
            if i == 0:
                s = s + ' ' + self.print_tree(self.tree[r][i], l)
            else:
                s = s + ' ' + self.print_tree(self.tree[r][i], l + '.' + str(i))
        if self.deg[r] > 0:
            s = s + ')'
        return s

    def add_child(self, par, ch):
        self.p[ch] = par
        self.tree[par][self.deg[par]] = ch
        self.deg[par] += 1

    def make_tree(self):
        par = 0
        for i in range(1, len(self.tokens)):
            if self.level[i] < self.level[i-1]:
                while self.level[par] > self.level[i]:
                    par = self.p[par]
                par = self.p[par]
            if self.level[i] > self.level[i-1]:
                par = i
            if self.tokens[i] == '/' or self.tokens[i][0] == ':':
                self.add_child(par, i)
                self.add_child(i, i+1)

    @staticmethod
    def find_first_of(string, chars, pos):
        char_set = set(list(chars))
        for i, string_char in enumerate(string[pos:]):
            if string_char in char_set:
                return i + pos
        return -1

    def parse(self, s0):
        v    = []   # string
        prev = 0
        #while (pos := self.find_first_of(s0, ' ()', prev)) < 0:   # := operator new in python 3.8
        while True:
            pos = self.find_first_of(s0, ' ()', prev)
            if pos < 0: break
            if pos > prev:
                if s0[prev] == '\"':
                    pos = prev + 1
                    while s0[pos] != '\"':
                        pos += 1
                    while s0[pos] != ')' and s0[pos] != '(' and s0[pos] != ' ':
                        pos += 1
                v.append(s0[prev:pos])
            if s0[pos] == '(':
                v.append('(')
            if s0[pos] == ')':
                v.append(')')
            prev = pos + 1
        if prev < len(s0):
            v.append(s0[prev:])
        self.tokens = []
        self.level = []
        l = 0
        for i in range(1, len(v)):
            if v[i] == '(':
                l += 1
                continue
            if v[i] == ')':
                l -= 1
                continue
            self.tokens.append(v[i])
            self.level.append(l)
        for i in range(1000):
            self.deg[i] = 0
            self.p[i] = -1
        self.make_tree()
