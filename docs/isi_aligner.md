# Information Sciences Institute Aligner

This is an algorithmic aligner based on the paper [Aligning English Strings with Abstract Meaning Representation Graphs](https://www.isi.edu/natural-language/mt/amr_eng_align.pdf).
The code is a python`ized version of the ISI aligner code, which is a bunch of bash scripts and a
few c++ files. A copy of that project can be found [here](https://github.com/melanietosik/string-to-amr-alignment).

Due to the complexity of the alignment process and the underlying mgiza aligner, the code is not
setup to be used as part of the library for inference.  If you are doing simple inference, it's
recommended that you use the [faa aligner](https://amrlib.readthedocs.io/en/latest/faa_aligner/).
If you want to use this code, expect to need to modify the scripts a bit and customize it for
your use case, as this is not setup for ease of use.

The ISI alignment code is included here because this is the aligner that has been commonly used
with AMR and, I believe, the aligner used to create alignments for LDC2020T02.  It also performs
slightly better than the FAA aligner (see performance at the bottom)

To use the code you will need to install and compile [mgiza](https://github.com/moses-smt/mgiza/tree/master/mgizapp).

Note that the main alignment process is a bash script so this will not run under Windows, though
it could be converted if someone wanted to put in the effort.


### Usage
There are no library calls associated with the aligner.  All of the code is in the scripts
directory under the [ISI Aligner](https://github.com/bjascob/amrlib/tree/master/scripts/62_ISI_Aligner).
These scripts are simply run in order to conduct the alignment and scoring process.  You will
need a copy of LDC2014T12 to run the code, although it could easily be modified to run on
other versions.  For scoring, the original AMR 1.0 corpus is required as the gold alignments are
tied to these graphs.

Directories and file locations are generally setup in each script under the `__main__` statement.
Note that you will need to set the location of the `mgiza` binaries at the top of the bash script
`Run_Aligner.sh`

Unlike neural net models, the mgiza aligner doesn't natively separate training and inference into
two distinct steps.  Training and alignment all happen as part of the same process.  While it is
possible to re-use the pretrained tables to do inference, the scores generally drop a few points
(possibly because it resumes training on the smaller inference dataset) and the code here is not
setup to do inference.

If you would like to align your own sentences / graphs, I would recommend modifying the script
`Gather_LDC.py` and having the code append them on to the `sents.txt` and `gstrings.txt` files
created by the  script.  The alignments can then be extracted from the end of the
`amr_alignment_strings.txt` file after running all all steps (scripts) of the process.


## Performance
Score of the ISI_Aligner against the gold ISI hand alignments for LDC2014T12
```
Dev scores    Precision: 93.78   Recall: 80.30   F1: 86.52
Test scores   Precision: 92.05   Recall: 76.64   F1: 83.64
```

Scores here resemble the scores from the original paper within normal run-to-run variation
of ~0.5 points.

These scores are obtained during training.  When scoring with only the test/dev sets and
using pre-trained parameters, the scores drop around 2-3 points.
