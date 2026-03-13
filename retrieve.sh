#!/bin/bash

if [ $# -lt 2 ]
then
        echo "Usage: $0 <index dir> <query file>"
        exit
fi

INDEX_DIR=$1
QUERY_FILE=$2

#mvn exec:java -Dexec.mainClass="retrieval.SupervisedRLM" -Dexec.args="$1"
mvn exec:java -Dexec.mainClass="retrieval.OneStepRetriever" -Dexec.args="$INDEX_DIR $QUERY_FILE 1000"
