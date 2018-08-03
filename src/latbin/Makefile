
all:

EXTRA_CXXFLAGS = -Wno-sign-compare
include ../kaldi.mk

BINFILES = lattice-lmrescore-cuedrnnlm

OBJFILES =



TESTFILES =

ADDLIBS = ../lat/kaldi-lat.a ../lm/kaldi-lm.a ../hmm/kaldi-hmm.a \
          ../tree/kaldi-tree.a ../util/kaldi-util.a ../matrix/kaldi-matrix.a \
          ../thread/kaldi-thread.a ../fstext/kaldi-fstext.a ../base/kaldi-base.a \
          ../cuedlmcpu/cued-lm.a

include ../makefiles/default_rules.mk

LDLIBS += -fopenmp

