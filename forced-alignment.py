#!/usr/bin/env python
# coding: utf-8
"""
Tool which aligns audio and transcript of [Plumcot data](https://github.com/hbredin/pyannote-db-plumcot) using vrbs.

Usage:
    forced-alignment.py preprocess <serie_uri> <plumcot_path> [--wav_path=<wav_path> --aligned_path=<aligned_path>]
    forced-alignment.py postprocess <serie_uri> <plumcot_path> <serie_split> [options]
    forced-alignment.py check_files <serie_uri> <plumcot_path> [--wav_path=<wav_path> --aligned_path=<aligned_path>]
    forced-alignment.py split_regions <file_path> [--threshold]
    forced-alignment.py update_RTTM <rttm_path> <uem_path> <json_path> <file_uri>
    forced-alignment.py update_aligned <aligned_path> <json_path> <file_uri>
    forced-alignment.py write_RTTM <json_path> <file_uri>
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
    <rttm_path>                             Output of postprocess
    <uem_path>                              Output of postprocess
    <json_path>                             Path to the manually corrected, gecko-compliant json
    <file_uri>                              uri of the file you corrected (should be in the RTTM file)
    <aligned_path>                          Output of postprocess

Common options:
    --aligned_path=<aligned_path>           Defaults to <plumcot_path>/Plumcot/data/<serie_uri>/forced-alignment

preprocess options:
    --transcripts_path=<transcripts_path>   Defaults to <plumcot_path>/Plumcot/data/<serie_uri>/transcripts
    --wav_path=<wav_path>                   Checks that all files in file_list.txt are in <wav_path>
                                            and vice-versa. Defaults to not checking.

postprocess options:
    --transcripts_path=<transcripts_path>   Defaults to <plumcot_path>/Plumcot/data/<serie_uri>/transcripts
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

split_regions options:
    <file_path>                             Absolute path to the gecko-json file you want to preprocess
    --threshold                             Duration of the silence (s) between two words so the region is split
                                            Defaults to 0.15 seconds.
"""

# # Dependencies

from docopt import docopt

#I/O
import json
import xml.etree.ElementTree as ET
import re
import os
from pathlib import Path

#Meta
from typing import TextIO,Union
import warnings

#utils
from utils import normalize_string, do_this
from convert import *

#pyannote
from pyannote.core import Annotation,Segment,Timeline,notebook,SlidingWindowFeature,SlidingWindow
from pyannote.database.util import load_rttm, load_uem

def write_brackets(SERIE_PATH,TRANSCRIPTS_PATH):
    """
    Puts brackets around the [speaker_id] (first token of each line of the script) in the scripts
    Also writes a file with all the file uris in SERIE_PATH/file_list.txt
    """
    file_counter=0
    file_list=[]
    for file_name in sorted(os.listdir(TRANSCRIPTS_PATH)):
        file_uri,extension=os.path.splitext(file_name)
        if extension==".txt":
            file_list.append(file_uri)#keep a list for qsub on m107

            #open file
            with open(os.path.join(TRANSCRIPTS_PATH,file_name),"r") as file:
                raw_script=file.read()

            #anonymyzes the script
            bracket_raw_script=""
            for speech_turn in raw_script.split("\n"):
                if speech_turn != '':
                    first_space=speech_turn.find(" ")
                    bracket_raw_script+="["+speech_turn[:first_space]+"]"+speech_turn[first_space:]+"\n"

            #writes back anonymized script .anonymous
            anonymous_path=os.path.join(TRANSCRIPTS_PATH,file_uri+".brackets")
            print("\rWriting file #{} to {}".format(file_counter,anonymous_path),end="")
            with open(anonymous_path,"w") as file:
                file.write(bracket_raw_script)
            file_counter+=1
    if file_counter==0:
        raise ValueError(f"no txt files were found in {TRANSCRIPTS_PATH}")
    with open(os.path.join(SERIE_PATH,"file_list.txt"),"w") as file:
        file.write("\n".join(file_list)    )
    print("\nsuccesfully wrote file list to",os.path.join(SERIE_PATH,"file_list.txt"))


def write_id_aligned(ALIGNED_PATH,TRANSCRIPTS_PATH):
    """
    writes json files as defined in functions xml_to_GeckoJSON and aligned_to_id
    """
    file_counter=0
    for file_name in os.listdir(ALIGNED_PATH):
        file_uri,extension=os.path.splitext(file_name)#file_uri should be common to xml and txt file
        if extension==".xml":
            with open(os.path.join(TRANSCRIPTS_PATH,file_uri+".txt"),"r") as file:
                raw_script=file.read()
            xml_tree=ET.parse(os.path.join(ALIGNED_PATH,file_name))
            xml_root = xml_tree.getroot()
            gecko_json=xml_to_GeckoJSON(xml_root,raw_script)
            json_path=os.path.join(ALIGNED_PATH,file_uri+".json")
            print("\rWriting file #{} to {}".format(file_counter,json_path),end="")
            file_counter+=1
            with open(json_path,"w") as file:
                json.dump(gecko_json,file,indent=4)
    if file_counter==0:
        raise ValueError(f"no xml files were found in {ALIGNED_PATH}")
    print()#new line for prettier print

def append_to_rttm(file: TextIO, output: Union[Timeline, Annotation]):
        """Write pipeline output to "rttm" file
        Parameters
        ----------
        file : file object
        output : `pyannote.core.Annotation`
            Pipeline output
        """
        warnings.warn("deprecated in favor of Annotation.write_rttm")
        if isinstance(output, Annotation):
            for s, t, l in output.itertracks(yield_label=True):
                line = (
                    f'SPEAKER {output.uri} 1 {s.start:.3f} {s.duration:.3f} '
                    f'<NA> <NA> {l} <NA> <NA>\n'
                )
                file.write(line)
            return

        msg = (
            f'Dumping {output.__class__.__name__} instances to "rttm" files '
            f'is not supported.'
        )
        raise NotImplementedError(msg)

def append_to_uem(file: TextIO, output: Timeline):
    """Write pipeline output to "uem" file
    Parameters
    ----------
    file : file object
    output : `pyannote.core.Timeline`
        Pipeline output
    """
    warnings.warn("deprecated in favor of Timeline.write_uem")
    if isinstance(output, Timeline):
        for segment in output:
            line = "{} 1 {} {}\n".format(
                output.uri,
                segment.start,
                segment.end
                )
            file.write(line)
        return

    msg = (
        f'Dumping {output.__class__.__name__} instances to "uem" files '
        f'is not supported.'
    )
    raise NotImplementedError(msg)

def gecko_JSONs_to_aligned(ALIGNED_PATH):
    file_counter=0
    for i,file_name in enumerate(sorted(os.listdir(ALIGNED_PATH))):
        uri,extension=os.path.splitext(file_name)#uri should be common to xml and txt file
        if extension==".json":
            print("\rprocessing file #{} from {}".format(file_counter,os.path.join(ALIGNED_PATH,file_name)),end="")
            file_counter+=1
            with open(os.path.join(ALIGNED_PATH,file_name),"r") as file:
                gecko_JSON=json.load(file)
            aligned=gecko_JSON_to_aligned(gecko_JSON,uri)
            with open(os.path.join(ALIGNED_PATH,uri+".aligned"),'w') as file:
                file.write(aligned)
    if file_counter==0:
        raise ValueError(f"no json files were found in {ALIGNED_PATH}")
    print("\ndone ;)")

def gecko_JSONs_to_RTTM(ALIGNED_PATH, ANNOTATION_PATH, ANNOTATED_PATH, serie_split,
    VRBS_CONFIDENCE_THRESHOLD =0.0, FORCED_ALIGNMENT_COLLAR=0.0,expected_min_speech_time=0.0):
    """
    Converts gecko_JSON files to RTTM using pyannote `Annotation`.
    Also keeps a track of files in train, dev and test sets.
    Also adds annotated parts of the files to a UEM depending on FORCED_ALIGNMENT_COLLAR.

    Parameters:
    -----------
    ALIGNED_PATH : path where gecko_JSON files are stored.
    ANNOTATION_PATH : path where to store the annotations in RTTM.
    ANNOTATED_PATH : path where to store the annotated parts of the files in UEM.
    VRBS_CONFIDENCE_THRESHOLD : `float`, the segments with confidence under VRBS_CONFIDENCE_THRESHOLD won't be added to UEM file.
        Defaults to 0.0
    FORCED_ALIGNMENT_COLLAR: `float`, Merge tracks with same label and separated by less than `FORCED_ALIGNMENT_COLLAR` seconds.
        Defaults to 0.0
    """
    if os.path.exists(ANNOTATION_PATH):
        raise ValueError("""{} already exists.
                         You probably don't wan't to append any more data to it.
                         If you do, remove this if statement.""".format(ANNOTATION_PATH))
    if os.path.exists(ANNOTATED_PATH):
        raise ValueError("""{} already exists.
                         You probably don't wan't to append any more data to it.
                         If you do, remove this if statement.""".format(ANNOTATED_PATH))
    file_counter=0
    train_list,dev_list,test_list=[],[],[]#keep track of file name used for train, dev and test sets
    for i,file_name in enumerate(sorted(os.listdir(ALIGNED_PATH))):
        uri,extension=os.path.splitext(file_name)#uri should be common to xml and txt file
        if extension==".json":
            print("\rprocessing file #{} from {}".format(file_counter,os.path.join(ALIGNED_PATH,file_name)),end="")
            #read file, convert to annotation and write rttm
            with open(os.path.join(ALIGNED_PATH,file_name),"r") as file:
                gecko_JSON=json.load(file)
            annotation,annotated=gecko_JSON_to_Annotation(gecko_JSON,uri,'speaker',
                VRBS_CONFIDENCE_THRESHOLD,FORCED_ALIGNMENT_COLLAR,
                expected_min_speech_time, manual=False)
            with open(ANNOTATION_PATH,'a') as file:
                annotation.write_rttm(file)
            with open(ANNOTATED_PATH,'a') as file:
                annotated.write_uem(file)
            #train dev or test ?
            season_number=int(re.findall(r'\d+', file_name.split(".")[1])[0])
            if season_number in serie_split["test"]:
                test_list.append(uri)
            elif season_number in serie_split["dev"]:
                dev_list.append(uri)
            elif season_number in serie_split["train"]:
                train_list.append(uri)
            else:
                raise ValueError("Expected season_number to be in {}\ngot {} instead".format(serie_split,season_number))
            file_counter+=1
    if file_counter==0:
        raise ValueError(f"no json files were found in {ALIGNED_PATH}")
    with open(os.path.join(SERIE_PATH,"train_list.lst"),"w") as file:
        file.write("\n".join(train_list))
    with open(os.path.join(SERIE_PATH,"dev_list.lst"),"w") as file:
        file.write("\n".join(dev_list))
    with open(os.path.join(SERIE_PATH,"test_list.lst"),"w") as file:
        file.write("\n".join(test_list))
    print("\nDone, succefully wrote the rttm file to {}\n and the uem file to {}".format(ANNOTATION_PATH,ANNOTATED_PATH))

def check_files(SERIE_PATH,wav_path,aligned_path):
    with open(os.path.join(SERIE_PATH,"file_list.txt"),'r') as file:
        file_list=set(file.read().split("\n"))
    with open(os.path.join(SERIE_PATH,"episodes.txt"),'r') as file:
        episodes=file.read().split("\n")
        episodes=set([episode.split(',')[0] for episode in episodes])
    if wav_path:
        wav_uris=[]
        for file_name in sorted(os.listdir(wav_path)):
            uri,extension=os.path.splitext(os.path.splitext(file_name)[0])
            if extension == '.en16kHz':
                wav_uris.append(uri)
        wav_uris=set(wav_uris)
        if file_list - wav_uris:
            warnings.warn(
                f'{sorted(file_list - wav_uris)} are not in {wav_path} '
                f'(but are in {SERIE_PATH}).'
            )
        if wav_uris - file_list:
            warnings.warn(
                f'{sorted(wav_uris - file_list)} are not in {SERIE_PATH} '
                f'(but are in {wav_path}).'
            )
    else:
        warnings.warn("--wav_path was not specified.")
    if aligned_path:
        aligned_uris=[]
        for file_name in sorted(os.listdir(aligned_path)):
            uri,extension=os.path.splitext(os.path.splitext(file_name)[0])
            if extension == '.xml':
                aligned_uris.append(uri)
        aligned_uris=set(aligned_uris)
        if file_list - aligned_uris:
            warnings.warn(
                f'{sorted(file_list - aligned_uris)} are not in {aligned_path} '
                f'(but are in {SERIE_PATH}).'
            )
        if aligned_uris - file_list:
            warnings.warn(
                f'{sorted(aligned_uris - file_list)} are not in {SERIE_PATH} '
                f'(but are in {aligned_path}).'
            )
    else:
        warnings.warn("--aligned_path was not specified.")
    if file_list - episodes:
        warnings.warn(
            f'{sorted(file_list - episodes)} are not in episodes.txt '
            f'(but are in {SERIE_PATH}).'
        )
    if episodes - file_list:
        warnings.warn(
            f'{sorted(episodes - file_list)} are not in {SERIE_PATH} '
            f'(but are in episodes.txt).'
        )
    print("Done checking files. (No warning means everything is okay.)")

def split_regions(file_path,threshold):
    with open(file_path,'r') as file:
        gecko_json=json.load(file)
    for i, monologue in enumerate(gecko_json['monologues']):
        terms=monologue["terms"]
        for j, term in enumerate(terms):
            if j+1 < len(terms):#next term exists
                if terms[j+1]['start']-term['end']>threshold:
                    #split monologue in two
                    new_monologue={
                        "speaker":monologue["speaker"],
                        "terms":terms[j+1:]
                    }
                    gecko_json['monologues'][i]["terms"]=terms[:j+1]
                    gecko_json['monologues'].insert(i+1,new_monologue)
                    break
    dir_path,file_name=os.path.split(file_path)
    file_uri,_=os.path.splitext(file_name)
    new_path=os.path.join(dir_path,f'{file_uri}.{threshold}.json')
    with open(new_path,'w') as file:
        json.dump(gecko_json,file,indent=4)
    print(f"succesfully dumped {new_path}")


def update_RTTM(rttm_path, uem_path, json_path, file_uri):
    if file_uri not in json_path:
        warnings.warn(f"replacing {file_uri} in RTTM by {json_path}")
    print("loading RTTM...")
    rttm=load_rttm(rttm_path)
    print("loading UEM...")
    uem=load_uem(uem_path)

    with open(json_path, 'r') as file:
        gecko_JSON=json.load(file)
    annotation,annotated=gecko_JSON_to_Annotation(gecko_JSON, file_uri, 'speaker', manual=True)
    rttm[file_uri]=annotation
    uem[file_uri]=annotated
    print("writing updated RTTM...")
    with open(rttm_path,'w') as file:
        for annotation in rttm.values():
            annotation.write_rttm(file)
    print("writing updated UEM...")
    with open(uem_path, 'w') as file:
        for annotated in uem.values():
            annotated.write_uem(file)
    print(f"succesfully dumped {rttm_path} and {uem_path}")

def write_RTTM(json_path, file_uri):
    rttm_path=Path(json_path.parent,f"{file_uri}.manual.rttm")
    uem_path=Path(json_path.parent,f"{file_uri}.manual.uem")
    with open(json_path, 'r') as file:
        gecko_JSON=json.load(file)
    annotation,annotated=gecko_JSON_to_Annotation(gecko_JSON, file_uri, 'speaker', manual=True)
    with open(rttm_path,'w') as file:
        annotation.write_rttm(file)
    with open(uem_path, 'w') as file:
        annotated.write_uem(file)
    print(f"succesfully dumped {rttm_path} and {uem_path}")

def update_aligned(aligned_path, json_path, file_uri):
    if file_uri not in json_path:
        warnings.warn(f"replacing {aligned_path} by {json_path}")
    with open(json_path, 'r') as file:
        gecko_JSON=json.load(file)
    aligned=gecko_JSON_to_aligned(gecko_JSON,file_uri)
    with open(aligned_path,'w') as file:
        file.write(aligned)
    print(f"succesfully dumped {aligned_path}")

if __name__ == '__main__':
    args = docopt(__doc__)
    if args['split_regions']:
        file_path=args['<file_path>']
        threshold=float(args["--threshold"]) if args["--threshold"] else 0.15
        split_regions(file_path,threshold)
    elif args['update_RTTM']:
        rttm_path=args['<rttm_path>']
        uem_path=args['<uem_path>']
        json_path=args['<json_path>']
        file_uri=args['<file_uri>']
        update_RTTM(rttm_path, uem_path, json_path, file_uri)
    elif args['update_aligned']:
        aligned_path=args['<aligned_path>']
        json_path=args['<json_path>']
        file_uri=args['<file_uri>']
        update_aligned(aligned_path, json_path, file_uri)
    elif args['write_RTTM']:
        json_path=Path(args['<json_path>'])
        file_uri=args['<file_uri>']
        write_RTTM(json_path,file_uri)
    else:
        serie_uri=args["<serie_uri>"]
        plumcot_path=args["<plumcot_path>"]
        SERIE_PATH=os.path.join(plumcot_path,"Plumcot","data",serie_uri)
        transcripts_path=args["--transcripts_path"] if args["--transcripts_path"] else os.path.join(SERIE_PATH,"transcripts")
        aligned_path = args["--aligned_path"] if args["--aligned_path"] else os.path.join(SERIE_PATH,"forced-alignment")
        if args['check_files']:
            wav_path=os.path.join(args['--wav_path'],serie_uri) if args['--wav_path'] else None
            check_files(SERIE_PATH,wav_path,aligned_path)
        elif args['preprocess']:
            print("adding brackets around speakers id")
            write_brackets(SERIE_PATH,transcripts_path)
            print("done, you should now launch vrbs before converting")
            wav_path=os.path.join(args['--wav_path'],serie_uri) if args['--wav_path'] else None
            if not os.path.exists(aligned_path):
                os.mkdir(aligned_path)
            check_files(SERIE_PATH,wav_path,aligned_path=None)
        elif args['postprocess']:
            serie_split={}
            for key, set in zip(["test","dev","train"],args["<serie_split>"].split(",")):
                serie_split[key]=list(map(int,set.split("-")))
            expected_min_speech_time=float(args["--expected_time"]) if args["--expected_time"] else float('inf')
            vrbs_confidence_threshold=float(args["--conf_threshold"]) if args["--conf_threshold"] else 0.0
            forced_alignment_collar=float(args["--collar"]) if args["--collar"] else 0.0
            annotation_path=os.path.join(aligned_path,"{}_{}collar.rttm".format(serie_uri,forced_alignment_collar))
            annotated_path=os.path.join(aligned_path,"{}_{}confidence.uem".format(serie_uri,vrbs_confidence_threshold))

            print("converting vrbs.xml to vrbs.json and adding proper id to vrbs alignment")
            write_id_aligned(aligned_path,transcripts_path)
            if do_this("Would you like to convert annotations from gecko_JSON to RTTM ?"):
                gecko_JSONs_to_RTTM(aligned_path, annotation_path, annotated_path, serie_split,
            else:
                print("Okay, no hard feelings")
            if do_this("Would you like to convert annotations from gecko_JSON to LIMSI-compliant 'aligned' ?"):
                gecko_JSONs_to_aligned(aligned_path)
            else:
                print("Okay then you're done ;)")
