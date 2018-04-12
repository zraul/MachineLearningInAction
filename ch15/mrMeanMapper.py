# coding:utf-8

import sys
from numpy import *

def read_line(file):
    for line in file:
        yield line.rstrip()

input = read_line(sys.stdin)
input = [float(line) for line in input]
numInputs = len(input)
input = mat(input)
sqInput = power(input, 2)

print '%d\t%f\t%f' % (numInputs, mean(input), mean(sqInput))
print >> sys.stderr, 'repost:still alive'