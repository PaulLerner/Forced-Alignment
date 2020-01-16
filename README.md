# Forced-Alignment
Tool which aligns audio and transcript of [Plumcot data](https://github.com/hbredin/pyannote-db-plumcot) using vrbs.

## Main

```
Usage:
    forced-alignment.py preprocess <serie_uri> <plumcot_path> [--wav_path=<wav_path>]
    forced-alignment.py postprocess <serie_uri> <plumcot_path> <serie_split> [options]
    forced-alignment.py -h | --help

Arguments:
    <serie_uri>                             One of Plumcot/data/series.txt
    <plumcot_path>                          something like /path/to/pyannote-db-plumcot
    <serie_split>                           <test>,<dev>,<train> where <test>, <dev> and <train>
                                            are seasons number separated by '-' that should be in the data subset
                                            e.g. : 1,2-3,4-5-6-7-8-9-10
    <wav_path>                              a priori /vol/work3/maurice/dvd_extracted/
                                            should contain a folder named <serie_uri>
                                            itself containg plenty of wav files

preprocess options:
    --transcripts_path=<transcripts_path>   Defaults to <plumcot_path>/Plumcot/data/<serie_uri>/transcripts
    --wav_path=<wav_path>                   Checks that all files in file_list.txt are in <wav_path>
                                            and vice-versa. Defaults to not checking.

postprocess options:
    --transcripts_path=<transcripts_path>   Defaults to <plumcot_path>/Plumcot/data/<serie_uri>/transcripts
    --aligned_path=<aligned_path>           Defaults to <plumcot_path>/Plumcot/data/<serie_uri>/forced-alignment
    --expected_time=<expected_time>         `float`, Optional.
                                            Threshold (in seconds) under which the total duration of speech time
                                            is suspicious (warns the user).
                                            Defaults to never suspect anything (i.e. +infinity)
                                            Recommended : 200.0
    --conf_threshold=<conf_threshold>       `float`, the segments with confidence under `conf_threshold`
                                            won't be added to UEM file.
                                            Defaults to 0.0
                                            Recommended : 0.5
    --collar=<collar>                       `float`, Merge tracks with same label and separated by less than `collar` seconds.
                                            Defaults to 0.0
                                            Recommended : 0.15
```

### Preprocessing (`preprocess`)
`forced-alignment.py preprocess` first adds brackets around normalized-characters names in the scripts defined in [pyannote.db](https://github.com/hbredin/pyannote-db-plumcot/blob/develop/CONTRIBUTING.md#idepisodetxt)

e.g. :
```bash
./forced-alignment.py preprocess Friends /vol/work/lerner/pyannote-db-plumcot \
--wav_path=/vol/work3/maurice/dvd_extracted
```

### Actual forced-alignment - VRBS (`forced-alignment.sh`)

You should then launch `forced-alignment.sh` to align audio and transcription. Unfortunately, it requires vrbs which is closed source. Note that it cannot handle files more than 200 MB large.

Usage :
```bash
forced-alignment.sh /path/to/pyannote-db-plumcot/Plumcot/data/${SERIE_URI}/file_list.txt ${SERIE_URI} /path/to/pyannote-db-plumcot
```

Usage with m107 (see [cluster 101](http://herve.niderb.fr/cluster101/)) :
```bash
export SERIE_URI=Friends
export N_FILES=231 #number of lines in file_list.txt
export LOGS=/vol/work/lerner/logs #defaults to ~
qsub -tc 10 -t 1-${N_FILES} -o ${LOGS} -e ${LOGS} forced-alignment.sh /vol/work/lerner/pyannote-db-plumcot/Plumcot/data/${SERIE_URI}/file_list.txt ${SERIE_URI} /vol/work/lerner/pyannote-db-plumcot
```

### Post-processing (`postprocess`)

Once vrbs is done you can continue with `forced-alignment.py postprocess` which will transform the XML output of vrbs into [Gecko](https://github.com/gong-io/gecko) compliant-JSON. The file formats are described below. The script also removes speakers id from the transcript and puts them instead in a proper JSON attribute : `speaker["id"]`.

*About `<serie_split>`* : The test set should always be the first season, the dev set might be season 2 or 2 and 3 depending on the data size.

e.g. :
```bash
./forced-alignment.py postprocess Friends /vol/work/lerner/pyannote-db-plumcot \
1,2-3,4-5-6-7-8-9-10 --expected_time=200 --conf_threshold=0.5 --collar=0.15
```





After that, you may or may not want to convert all the annotations from gecko_JSON to RTTM, this relies on pyannote.core.

Type "n" or "no" (case insensitive) if you don't want to.

*You're done !*

## Manual correction
### Pre-processing for gecko (`split_regions`)

If you plan on correcting the errors of the forced-alignment using gecko, you might want to use `split_regions` before-hand. So that the segment timings are more accurate.

```
Usage:
    forced-alignment.py split_regions <file_path> [--threshold]
    forced-alignment.py -h | --help

split_regions options:
    <file_path>                             Absolute path to the gecko-json file you want to preprocess
    --threshold                             Duration of the silence (s) between two words   so the region is split
```

e.g. :

`./forced-alignment.py split_regions /vol/work/lerner/pyannote-db-plumcot/Plumcot/data/Friends/forced-alignment/Friends.Season01.Episode01.json`

### Update RTTM (`update_RTTM`)

Once you're done with correcting the json file in gecko, you might want to convert it to RTTM and aligned...

```
Usage:
    forced-alignment.py update_RTTM <rttm_path> <uem_path> <json_path> <file_uri>
    forced-alignment.py update_aligned <aligned_path> <json_path>
    forced-alignment.py -h | --help

Arguments:
    <rttm_path>                             Output of postprocess
    <uem_path>                              Output of postprocess
    <json_path>                             Path to the manually corrected, gecko-compliant json
    <file_uri>                              uri of the file you corrected (should be in the RTTM file)
```

### Update aligned (`update_aligned`)

```
Usage:
    forced-alignment.py update_aligned <aligned_path> <json_path> <file_uri>
    forced-alignment.py -h | --help

Arguments:
    <aligned_path>                          Output of postprocess
    <json_path>                             Path to the manually corrected, gecko-compliant json
    <file_uri>                              uri of the file you corrected (should match aligned_path)
```

# Format
## XML (VRBS)
```py
AudioDoc  
└───ProcList    #0  
│   ChannelList #1  
│   SpeakerList #2  
│   SegmentList #3  (list)  
│   └───SpeechSegment #(dict[attrib], list[Word])  
│   │   └───Word
│   │   │   └───id #(unique for all words, regardless of speechsegment)
│   │   │   │   stime #(start time in seconds)
│   │   │   │   dur #(duration in seconds)
│   │   │   │   conf #(confidence between 0 and 1)
│   │   │   spkid #(speaker id, could be used as name or vrbs_id)
```

## JSON (Gecko)

```py
Root
└───monologues #(list of speakers)
│   └───monolog_0 #(1st speech turn)
│   │   └──speaker #(1st speech turn's speaker)
│   │   │  terms #(list of terms)
│   │   │   └──term_0 #(1st speech turn's 1st term)
│   │   │   │  └──start #(1st speech turn's 1st term start time in SECONDS)
│   │   │   │  │  confidence #(1st speech turn's 1st term start confidence (between 0.0 and 1.0))
│   │   │   │  │  text #str: content of the word
```

## Aligned (LIMSI)
Inspired by [`stm`](http://www1.icsi.berkeley.edu/Speech/docs/sctk-1.2/infmts.htm#stm_fmt_name_0) the `aligned` format provides additionally the confidence of the model in the transcription :

```
<file_uri> <speaker_id> <start_time> <end_time> <token> <confidence_score>
```
e.g. :
```
TheBigBangTheory.Season01.Episode01 sheldon_cooper start_time end_time How 0.9
TheBigBangTheory.Season01.Episode01 sheldon_cooper start_time end_time are 0.6
TheBigBangTheory.Season01.Episode01 sheldon_cooper start_time end_time you 0.8
TheBigBangTheory.Season01.Episode01 sheldon_cooper start_time end_time , 0.1
TheBigBangTheory.Season01.Episode01 sheldon_cooper start_time end_time Leonard 0.5
TheBigBangTheory.Season01.Episode01 sheldon_cooper start_time end_time ? 0.2
```
