#!/usr/bin/env bash

base_url="https://zenodo.org/records/4940267/files"

for i in $(seq 1 79); do
  fname="eeg${i}.edf"
  url="${base_url}/${fname}?download=1"
  echo "Downloading ${fname}..."
  curl -L -o "${fname}" "${url}"
done

echo "All downloads complete."
