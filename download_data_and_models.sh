#!/bin/bash

curl -L -o datasets.tar.gz https://nextcloud.cs.uwaterloo.ca/s/iZRD2SdgjKZRskR/download
curl -L -o pretrained_models.tar.gz.aa https://nextcloud.cs.uwaterloo.ca/s/tCdKk8x8z2na4mA/download
curl -L -o pretrained_models.tar.gz.ab https://nextcloud.cs.uwaterloo.ca/s/7XMBndAdPBSwBNE/download
curl -L -o pretrained_models.tar.gz.ac https://nextcloud.cs.uwaterloo.ca/s/kpNFGddk2tbK2Hn/download
curl -L -o pretrained_models.tar.gz.ad https://nextcloud.cs.uwaterloo.ca/s/FEnfHBQctWMLYKE/download
curl -L -o pretrained_models.tar.gz.ae https://nextcloud.cs.uwaterloo.ca/s/swW6PT4kXx3W3HN/download
curl -L -o pretrained_models.tar.gz.af https://nextcloud.cs.uwaterloo.ca/s/8Q9yPd94kRKfXgf/download
curl -L -o pretrained_models.tar.gz.ag https://nextcloud.cs.uwaterloo.ca/s/jZjNMC4wdemccEf/download

tar -zxvf datasets.tar.gz
cat pretrained_models.tar.gz.* | tar xzvf -
rm -rf pretrained_models.tar.gz.*
rm -rf datasets.tar.gz
