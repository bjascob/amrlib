# Wiki Tags
The AMR spec includes `:wiki` tags for named-entities in a graph.  Since these are very domain
specific, and given the liminted LDC training data, it is problematic to try to teach the model
to apply these.  For this reason they are omitted completely from the trained model and instead,
can be applied as a post-process operation, using the
[DBPedia Spotlight](https://www.dbpedia-spotlight.org/) server.

Unfortunately the online server has been unreliable so if you wish to apply the tags you should
setup a local version of the server which is relatively easy and detailed below.

For now, the code omits adding these tags when using the GUI or the sequence-to-graph model but
the library does include these functions to do the lookup and add the tags if someone wants to
take the time to add the function calls.

## Local Server Setup Notes
* Download from... https://sourceforge.net/projects/dbpedia-spotlight/files/
* spotlight/dbpedia-spotlight-1.0.0.jar and 2016-10/en/model/en.tar.gz (dated 2018-02-18) 1.9GB
* tar xzf en.tar.gz
* Instructions at https://github.com/dbpedia-spotlight/dbpedia-spotlight/wiki/Run-from-a-JAR
  but the wget downloads didn't work.  Use the sourceforge download location above
* Note that this will not run with java 11, use java 8 instead

To run the server execute the command..

`java -jar dbpedia-spotlight-1.0.0.jar <model path>/en/ http://localhost:2222/rest`

There is a simple script in `scripts/10_Misc/SpotlightDBServer.sh` to do this


## Adding :wiki Tags to Graphs
Once the local server is setup and running, go to `scripts\30_Model_Parse_GSII` and open the script
`32_Add_Wiki.py`.  You can modify the in and out filenames as needed.

For additional details see `amrlib/graph_processing/wiki_adder.py`

## Accuracy
For the parse_gsii v0.1.0 model
```
Without wiki tags (so they don't contribute to the score at all)
Smatch           -> P: 0.790,  R: 0.746,  F: 0.767
With wiki tags added
Smatch           -> P: 0.784,  R: 0.740,  F: 0.762
Wikification     -> P: 0.732,  R: 0.632,  F: 0.678
```
Which shows that the lookup gets the tag correct about 2/3 of the time and has minimal
impact on the overall smatch score.
