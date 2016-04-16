#/usr/bin/env python2

from __future__ import print_function

from argparse import ArgumentParser
from scipy.signal import hanning
import numpy as np
from numpy.fft import fft, fftfreq
import scipy.io.wavfile
import matplotlib.pyplot as plt

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i+n]

def s_to_samples(samplerate, s):
    '''Convert second to samples'''
    return int(s * samplerate)

def sample_to_s(samplerate, sample):
    '''Convert sample number to second'''
    return sample/float(samplerate)

def generate_midi(infile, fraction):
    samplerate, data = scipy.io.wavfile.read(infile)

    ws = 0.5/fraction

    ns = s_to_samples(samplerate, ws)
    print(ns)

    w = hanning(ns)

    spectra = []

    for chunk in chunks(data, ns):
        delta = len(w) - len(chunk)
        if delta:
            chunk = np.append(chunk, [0] * delta)

        spectrum = fft(chunk * w)

        spectra.append(spectrum)
        # print(d)

    return spectra

if __name__ == '__main__':
    parser = ArgumentParser(description='generate midi from wav file')
    parser.add_argument('-i', '--infile',
                        default='in.wav',
                        help="input wav file (default is 'in.wav')")

    parser.add_argument('-f', '--fraction',
                        default=64,
                        help="fractional note (default = 64)")

    args = parser.parse_args()

    generate_midi(args.infile, args.fraction)
