#!/bin/bash

#$ -N forced-alignment
#$ -S /bin/bash
#$ -v LD_LIBRARY_PATH
#$ -v PATH

echo "This is task ${SGE_TASK_ID} of job ${JOB_ID}."
export VRBS_ROOT=/vol/jlg1/gauvain/vrbs
export VRBS_TMP=/usr/tmp
export VRBS_BIN=$VRBS_ROOT/bin
export VRBS_PART=$VRBS_ROOT/part
export VRBS_LID=$VRBS_ROOT/lid
export VRBS_TRANS=$VRBS_ROOT/trans
export VRBS_SID=$VRBS_ROOT/sid
export PATH=$PATH:$VRBS_BIN
export FILE_URI=`head -n ${SGE_TASK_ID} $1 | tail -n 1`
echo file uri : $FILE_URI
echo serie uri : $2
echo plumcot path : $3
`which vrbs_align` -f /vol/work3/lefevre/dvd_extracted/$2/$FILE_URI.en16kHz.wav -o $3/Plumcot/data/$2/forced-alignment/$FILE_URI.xml -leng -qs -v $3/Plumcot/data/$2/transcripts/$FILE_URI.brackets
