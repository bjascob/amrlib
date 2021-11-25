#!/bin/sh

#### Howto Setup ####
# Original notes on setup from https://github.com/dbpedia-spotlight/dbpedia-spotlight/wiki/Run-from-a-JAR,
# however the wget didn't work so use the following..
# Download from... https://sourceforge.net/projects/dbpedia-spotlight/files/ +..
#   dbpedia-spotlight-1.0.0.jar from spotlight directory
#   model from 2016-10/en/model (dated 2018-02-18) 1.9GB
#   The "services" jar is not needed
# tar xzf en.tar.gz
# java -jar dbpedia-spotlight-1.0.0.jar path/to/model/folder/en_2+2 http://localhost:2222/rest


# Crashes on query when running with java 11
JAVA=/usr/lib/jvm/jdk1.8.0_261/bin/java

# Chage to the directory with the jar and db files
cd /home/bjascob/Libraries/spotlight

$JAVA -jar dbpedia-spotlight-1.0.0.jar en/ http://localhost:2222/rest
