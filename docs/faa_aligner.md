# Fast_Align Algorithm Aligner

This is an algorithmic aligner based on the paper [Aligning English Strings with Abstract Meaning Representation Graphs](https://www.isi.edu/natural-language/mt/amr_eng_align.pdf).
The code is based on the ISI aligner code.  A copy of that project can be found [here](https://github.com/melanietosik/string-to-amr-alignment).
The project makes use of original pre/post-processing code but replaces the use of the [mgiza](https://github.com/moses-smt/mgiza/tree/master/mgizapp)
app with [fast_align](https://github.com/clab/fast_align).  The bash scripts have been converted to python and a new
"inference" step allows for pre-trained parameters to be used during run-time operation.

To use the code you will need to install and compile the C++ code for [fast_align](https://github.com/clab/fast_align).
The compile process will produce binaries for `fast_align` and `atools` in the same directory.
Put these in your path or you can set the environment variable `FABIN_DIR` to their directory.
The aligner/fast_align binaries work under both Windows and Linux.

The aligner comes with pre-trained parameters that are included in a tar.gz file in the project.
The first time the aligner is run, it will un-tar the files in `amrlib/data/model_aligner_faa/`.

If you'd like to train, or just test the aligner, see the scripts in the [FAA_Aligner scripts directory](https://github.com/bjascob/amrlib/tree/master/scripts/61_FAA_Aligner)
You can run these in order to create a new model and test it.  Each script will complete in just a few seconds.
Note that the scripts are setup to use LDC2014T12 (AMR-1), since these are what the test hand-alignments are made from.



### Usage
To use the aligner you should have a list of sentences and an amr graphs, in string format.


Example aligner usage
```
from amrlib.alignments.faa_aligner import FAA_Aligner
inference = FAA_Aligner()
amr_surface_aligns, alignment_strings = inference.align_sents(sents, graph_strings)
print(alignment_strings)
```
The code returns the original amr graphs with surface alignments added and a list of alignment strings in ISI (not JAMR) format.

!! Note that the input `sents` need to be space tokenized strings.


## Performance
Score of the FAA_Aligner against the gold ISI hand alignments for LDC2014T12 <sup>**1</sup>
```
Dev scores    Precision: 89.30   Recall: 78.20   F1: 83.38
Test scores   Precision: 86.03   Recall: 79.00   F1: 82.37
```

<sup>**1</sup>
Note that these scores are obtained during training.  When scoring with only the test/dev sets and
using pre-trained parameters, the scores vary slightly (less than 0.5) from the original.
