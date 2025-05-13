#!/bin/bash

data_dir='./EMNIST_dataset'
curl https://biometrics.nist.gov/cs_links/EMNIST/gzip.zip --output "$data_dir/gzip.zip" 
FILES="emnist-balanced-train-images-idx3-ubyte.gz
       emnist-balanced-test-images-idx3-ubyte.gz
       emnist-balanced-train-labels-idx1-ubyte.gz
       emnist-balanced-test-labels-idx1-ubyte.gz"

for f in $FILES
do
 unzip -j "$data_dir/gzip.zip" "gzip/$f" -d $data_dir
 gzip -d $data_dir'/'$f
done
