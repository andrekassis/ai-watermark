#!/bin/bash

#Verify billing is enabled: "https://cloud.google.com/billing/docs/how-to/verify-billing-enabled#confirm_billing_is_enabled_on_a_project"
#Enable the API: "https://console.cloud.google.com/flows/enableapi?apiid=aiplatform.googleapis.com"

X86_64="https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-linux-x86_64.tar.gz"
ARM="https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-linux-arm.tar.gz"
X86="https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-linux-x86.tar.gz"

platform=`uname -a`
if [[ ! -z $(echo "$platform" | grep "x86_64") ]]; then
    link="$X86_64"
elif [[ ! -z $(echo "$platform" | grep "x86") ]]; then
    link="$X86"
else
    link="$ARM"
fi
#curl -O "$link"
file=`echo $link | rev | cut -f1 -d"/" | rev`
#tar -xf "$file"
./google-cloud-sdk/install.sh
source ~/.bashrc
hash -r
./google-cloud-sdk/bin/gcloud init
source ~/.bashrc
hash -r
gcloud auth application-default login

proj=`gcloud info | grep -Eo "Project: [^ ]*$" | cut -f2 -d" " | grep -v ^$ | head -1 | cut -c2- | rev | cut -c2- | rev`
region=`gcloud info | grep -Eo "region: .* \(property file\)" | cut -f2 -d" " | grep -v ^$ | head -1 | cut -c2- | rev | cut -c2- | rev`
sed -i "s/project_id:/project_id: ${proj}/g" systems/configs/synthid.yaml
sed -i "s/service:/service: ${region}/g" systems/configs/synthid.yaml

### Install Python SDK
#old SDK
pip install vertexai
pip install protobuf==3.20.0
echo "export SYNTHID_EXT=1" >> ~/.bashrc
source ~/.bashrc
hash -r

#if the old SDK doesn't work, use new API:
#pip install --upgrade google-genai
#export GOOGLE_CLOUD_PROJECT=$proj
#export GOOGLE_CLOUD_LOCATION=global
#export GOOGLE_GENAI_USE_VERTEXAI=True
