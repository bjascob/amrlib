MGIZA_BIN='/home/bjascob/Libraries/mgizapp/build/inst'    # absolute directory location here

THISDIR="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
SCRIPT_DIR=$THISDIR/../../amrlib/alignments/isi_aligner
CONFIG_DIR=$THISDIR/config

# change to the working directory
cd ../../amrlib/data/working_isi_aligner

# Transpose the output tables
function transpose_tables {
mv ttable.prev t0
cp *.t3.final t1
cp *.a3.final a1
cp *.d3.final d1
$SCRIPT_DIR/transpose-ttable.py 't0' 't1' 'ttable'
$SCRIPT_DIR/transpose-adtable.py 'a1' 'd1' 'atable' 'dtable'
rm t0 t1 a1 d1
mv ntable.prev ntable
mv *.t3.final ttable.prev
mv *.a3.final atable.prev
mv *.d3.final dtable.prev
mv *.n3.final ntable.prev
}


# plain2snt generate vcb (vocabulary) files and snt (sentence) files, containing the list of vocabulary and aligned sentences
# mkcls is a program to automatically infer word classes from a corpus using a maximum likelihood criterion (model 4 and 5 only)
# Generate for forward
SRC_FILE=`ls fr`
TRG_FILE=`ls en`
$MGIZA_BIN/plain2snt $SRC_FILE $TRG_FILE
$MGIZA_BIN/mkcls -p$SRC_FILE -V$SRC_FILE.vcb.classes
$MGIZA_BIN/mkcls -p$TRG_FILE -V$TRG_FILE.vcb.classes
$MGIZA_BIN/snt2cooc "$SRC_FILE"_"$TRG_FILE".cooc $SRC_FILE.vcb $TRG_FILE.vcb "$SRC_FILE"_"$TRG_FILE".snt

# Generate for backward
SRC_FILE=`ls en`
TRG_FILE=`ls fr`
$MGIZA_BIN/plain2snt $SRC_FILE $TRG_FILE
$MGIZA_BIN/mkcls -p$SRC_FILE -V$SRC_FILE.vcb.classes
$MGIZA_BIN/mkcls -p$TRG_FILE -V$TRG_FILE.vcb.classes
$MGIZA_BIN/snt2cooc "$SRC_FILE"_"$TRG_FILE".cooc $SRC_FILE.vcb $TRG_FILE.vcb "$SRC_FILE"_"$TRG_FILE".snt

# Run 5 each of HMM, Model-1, Model-4 forward  (en->fr)
rm ???-??-??.*
$MGIZA_BIN/mgiza  $CONFIG_DIR/giza.config.0.f >> log.txt
cp *.A3.final.* giza-align.txt
cp *.a3.final atable.prev
cp *.t3.final ttable.prev
cp *.hhmm.5 htable
cp *.hhmm.5.alpha htable.alpha
cp *.hhmm.5.beta htable.beta
cp *.d3.final dtable.prev
cp *.n3.final ntable.prev

# Run 5 each of HMM, Model-1, Model-4 backward (fr->en)
rm ???-??-??.*
$MGIZA_BIN/mgiza  $CONFIG_DIR/giza.config.0.b >> log.txt
cp *.hhmm.5 htable2
cp *.hhmm.5.alpha htable2.alpha
cp *.hhmm.5.beta htable2.beta

for i in {1..2}
do

# Run 5 iterations of Model 4 only forward (en->fr)
# restart=10, previous tables = ttable, attable, dtable, ntable, htable
transpose_tables
rm ???-??-??.*
$MGIZA_BIN/mgiza  $CONFIG_DIR/giza.config.f >> log.txt
transpose_tables

# Run 5 iterations of Model 4 only backward (fr-en)
# restart=10, previous tables = ttable, attable, dtable, ntable, htable2
# NOTE - uses previous htable2 instead of htable in above config
rm ???-??-??.*
$MGIZA_BIN/mgiza  $CONFIG_DIR/giza.config.b >> log.txt

done

# Do 5 iterations of Model 4 forward (en->fr)
# restart=10, previous tables = ttable, attable, dtable, ntable, htable
transpose_tables
rm ???-??-??.*
$MGIZA_BIN/mgiza  $CONFIG_DIR/giza.config.f >> log.txt

cp *.A3.final.* model_out.txt

# Cleanup
rm ???-??-??.*
