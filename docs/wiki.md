# Wiki Tags
The AMR spec includes `:wiki` tags for named-entities in a graph.  Since these are very domain
specific, and given the liminted LDC training data, it is problematic to try to train the model
to apply these.  For this reason they are omitted completely from the trained model and instead,
can be applied as a post-process operation. The library provides two different methods to do this,
BLINK and Spotlight.

The BLINK library provides higher performance over Spotlight. The following is a comparison of the
wikification results from the parse_t5-v0_2_0 model, looking at the overall smatch score for the
model and the enhanced score for wikification only.
```
  Blink:
        Smatch       -> P: 0.802,  R: 0.835,  F: 0.818
        Wikification -> P: 0.807,  R: 0.855,  F: 0.830
  Spotlight:
        Smatch       -> P: 0.798,  R: 0.830,  F: 0.813
        Wikification -> P: 0.687,  R: 0.758,  F: 0.721
```

The amrlib code omits adding the `:wiki` tags when using the GUI or the sequence-to-graph model function calls.
The functions to add the wiki tags are done as a post-process operation for evaluation after training
only.


## The BLINK Wikifier
The best performing wikification code is from [BLINK](https://github.com/facebookresearch/BLINK).  The code can be found at `amrlib/graph_processing/wiki_adder_blink.py`.  For an example
of how to use it see [amrlib/scripts/31_Model_Parse_T5/30_Add_Wiki_With_Blink.py](https://github.com/bjascob/amrlib/blob/master/scripts/31_Model_Parse_T5/30_Add_Wiki_With_Blink.py).

!! Note that the model for this code is more than 30GB and requires cuda and a GPU to run.

To use the BLINK system follow these setup instructions.
```
git clone https://github.com/facebookresearch/BLINK.git
cd BLINK
pip install -r requirements.txt
sh download_blink_models.sh
```
In the example code there are locations to specify the model directory and the library's location since
there is no pip install for BLINK.


## The Spotlight Wikifier
The [DBPedia Spotlight](https://www.dbpedia-spotlight.org/) server is the original solution amrlib used.
It is still potentially easier but there are a few issues with it's use.

Unfortunately the online server has been unreliable so if you wish to apply the tags you should
setup a local version of the server which is relatively easy and detailed below.

### Local Server Setup Notes
* Download from... https://sourceforge.net/projects/dbpedia-spotlight/files/
* spotlight/dbpedia-spotlight-1.0.0.jar and 2016-10/en/model/en.tar.gz (dated 2018-02-18) 1.9GB
* tar xzf en.tar.gz
* Instructions at https://github.com/dbpedia-spotlight/dbpedia-spotlight/wiki/Run-from-a-JAR
  but the wget downloads didn't work.  Use the sourceforge download location above
* Note that this will not run with java 11, use java 8 instead

To run the server execute the command..

`java -jar dbpedia-spotlight-1.0.0.jar <model path>/en/ http://localhost:2222/rest`

There is a simple script in `scripts/10_Misc/SpotlightDBServer.sh` to do this
