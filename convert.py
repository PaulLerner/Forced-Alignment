#utils
import json
import re
import os
from typing import TextIO,Union
import warnings

#pyannote
from pyannote.core import Annotation,Segment,Timeline,notebook,SlidingWindowFeature,SlidingWindow

def xml_to_GeckoJSON(xml_root,raw_script):
    """
    Parameters:
        xml_root : root of the xml tree defined by vrbs for forced alignment. root[3] should be SegmentList, a list of speech segments
        raw_script : `str` : the script as defined in https://github.com/hbredin/pyannote-db-plumcot/blob/develop/CONTRIBUTING.md#idepisodetxt
            Each line is a speech turn and the first (space-separated) token is the normalized speaker id.
    Returns:
        gecko_json : a JSON `dict` based on the demo file of https://github.com/gong-io/gecko/blob/master/samples/demo.json
            should be written to a file using json.dump"""
    gecko_json=json.loads("""{
      "schemaVersion" : "2.0",
      "monologues" : [  ]
    }""")
    gecko_json["monologues"]=[[] for _ in raw_script.split("\n")]
    json_i=0
    terms=[]
    current_speaker=xml_root[3][0][0].text.strip()[1:-1]
    for i,speech_segment in enumerate(xml_root[3]):
        for word in speech_segment:
            if word.text.strip()[0]=="[":#speaker id -> add new speaker
                speaker={
                "name" : None,
                "id" : current_speaker,#first and last charcater should be []
                "vrbs_id" : speech_segment.attrib['spkid']
                    }
                current_speaker=word.text.strip()[1:-1]
                gecko_json["monologues"][json_i]={
                    "speaker":speaker,
                    "terms":terms
                    }
                json_i+=1
                terms=[]
            else:
                terms.append(
                {
                    "start" : float(word.attrib['stime']),
                    "end" : float(word.attrib['stime'])+float(word.attrib['dur']),
                    "text" : word.text,
                    "type" : "WORD",
                    "confidence": float(word.attrib['conf'])
                })
    speaker={
    "name" : None,
    "id" : current_speaker,#first and last charcater should be []
    "vrbs_id" : speech_segment.attrib['spkid']
        }
    new_monologue={
            "speaker":speaker,
            "terms":terms
            }
    if json_i<len(gecko_json["monologues"]):
        gecko_json["monologues"][json_i]=new_monologue
    else:
        gecko_json["monologues"].append(new_monologue)
    gecko_json["monologues"].pop(0)

    return gecko_json

def gecko_JSON_to_aligned(gecko_JSON, uri=None):
    """
    Parameters:
    -----------
    gecko_JSON : `dict`
        loaded from a Gecko-compliant JSON as defined in xml_to_GeckoJSON
    uri (uniform resource identifier) : `str`
        which identifies the annotation (e.g. episode number)
        Defaults to None.

    Returns:
    --------
    aligned: `str`
        as defined in README one file per space-separated token.
        <file_uri> <speaker_id> <start_time> <end_time> <token> <confidence_score>
    """
    aligned=""
    for monologue in gecko_JSON["monologues"]:
        speaker_ids=monologue["speaker"]["id"].split("@")#defined in https://github.com/hbredin/pyannote-db-plumcot/blob/develop/CONTRIBUTING.md#idepisodetxt

        for i,term in enumerate(monologue["terms"]):
            for speaker_id in speaker_ids:#most of the time there's only one
                if speaker_id!='':#happens with "all@"
                    aligned+=f'{uri} {speaker_id} {term["start"]} {term["end"]} {term["text"].strip()} {term["confidence"]}\n'
    return aligned

def gecko_JSON_to_Annotation(gecko_JSON,uri=None,modality='speaker',confidence_threshold=0.0,collar=0.0,expected_min_speech_time=float("inf")):
    """
    Parameters:
    -----------
    gecko_JSON : `dict` loaded from a Gecko-compliant JSON as defined in xml_to_GeckoJSON
    uri (uniform resource identifier) : `str` which identifies the annotation (e.g. episode number)
        Default : None
    modality : `str` modality of the annotation as defined in https://github.com/pyannote/pyannote-core
    confidence_threshold : `float`, Optional. The segments with confidence under confidence_threshold won't be added to UEM file.
        Defaults to keep every segment (i.e. 0.0)
    collar: `float`, Optional. Merge tracks with same label and separated by less than `collar` seconds.
        Defaults to keep tracks timeline untouched (i.e. 0.0)
    expected_min_speech_time: `float`, Optional. Threshold (in seconds) under which the total duration of speech time is suspicious (warns the user).
        Defaults to never suspect anything (i.e. +infinity)
    Returns:
    --------
    annotation: pyannote `Annotation` for speaker identification/diarization as defined in https://github.com/pyannote/pyannote-core
    annotated: pyannote `Timeline` representing the annotated parts of the gecko_JSON files (depends on confidence_threshold)
    """
    annotation= Annotation(uri,modality)
    not_annotated=Timeline(uri=uri)
    total_speech_time=0.0
    for monolog in gecko_JSON["monologues"]:
        speaker_ids=monolog["speaker"]["id"].split("@")#defined in https://github.com/hbredin/pyannote-db-plumcot/blob/develop/CONTRIBUTING.md#idepisodetxt

        for i,term in enumerate(monolog["terms"]):
            for speaker_id in speaker_ids:#most of the time there's only one
                if speaker_id!='':#happens with "all@"
                    annotation[Segment(term["start"],term["end"]),speaker_id]=speaker_id
                    total_speech_time+=term["end"]-term["start"]
            if term["confidence"] <= confidence_threshold:
                not_annotated.add(Segment(term["start"],term["end"]))
    if total_speech_time<expected_min_speech_time:
        warnings.warn(f"total speech time of {uri} is only {total_speech_time})")
    return annotation.support(collar),not_annotated.gaps(support=Segment(0.0,term["end"]))
