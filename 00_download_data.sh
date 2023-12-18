#!/bin/bash

# download and process ESLO2
if [ ! -d $SCRATCH/eslo2_processed ]
then
    echo "Downloading ESLO2"
    if [ ! -d $SCRATCH/eslo2 ]
    then
        wget -O $SCRATCH/eslo2.zip https://repository.ortolang.fr/api/content/export?&path=/eslo/3/&filename=eslo2&scope=YW5vbnltb3Vz1
        unzip $SCRATCH/eslo2.zip -d $SCRATCH
    fi

    echo "Preparing ESLO2"
    python3 -m eslo2_prepare $SCRATCH/eslo2 $SCRATCH/eslo2_processed
fi
 
# download MUSAN RIRs
if [ ! -d $SCRATCH/RIRS_NOISES ]
then
    echo "Downloading MUSAN NOISE RIRs"
    wget -O $SCRATCH/rirs_noises.zip http://www.openslr.org/resources/28/rirs_noises.zip
    unzip $SCRATCH/rirs_noises.zip "RIRS_NOISES/pointsource_noises/*" -d $SCRATCH
fi

# download and process OpenSubtitles 2018 fr
if [ ! -d $SCRATCH/OpenSubtitles_processed ]
then
    echo "Downloading OpenSubtitles"
    if [ ! -d $SCRATCH/OpenSubtitles ]
    then
        wget -O $SCRATCH/OpenSubtitlesv2018fr.zip https://opus.nlpl.eu/download.php\?f\=OpenSubtitles/v2018/xml/fr.zip
        unzip $SCRATCH/OpenSubtitlesv2018fr.zip -d $SCRATCH
    fi

    echo "Preparing OpenSubtitles"
    python3 -m opensubtitles_prepare $SCRATCH/OpenSubtitles $SCRATCH/OpenSubtitles_processed
fi

# download CommonVoice 10.0 fr
if [ ! -d $SCRATCH/cv-corpus-10.0-2022-07-04 ]
then
    echo "Downloading CommonVoice"
    wget -O $SCRATCH/cv-corpus-10.0-2022-07-04-fr.tar.gz https://mozilla-common-voice-datasets.s3.dualstack.us-west-2.amazonaws.com/cv-corpus-10.0-2022-07-04/cv-corpus-10.0-2022-07-04-fr.tar.gz
    tar vxzf $SCRATCH/cv-corpus-10.0-2022-07-04-fr.tar.gz -C $SCRATCH
fi

# download PXSLU
if [ ! -d $SCRATCH/pxslu ]
then
    echo "Downloading PXSLU"
    wget -O $SCRATCH/pxslu.zip https://zenodo.org/record/6524162/files/pxslu.zip?download=1
    unzip $SCRATCH/pxslu.zip -d $SCRATCH/pxslu
    unzip $SCRATCH/pxslu/recordings.zip -d $SCRATCH/pxslu/recordings
fi

# download simsamu
if [ ! -d $SCRATCH/simsamu ]
then 
    echo "Downloading simsamu"
    cd $SCRATCH
    git clone --single-branch https://hf.co/datasets/medkit/simsamu
fi
