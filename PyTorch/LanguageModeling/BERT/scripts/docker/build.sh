#!/bin/bash
docker build . --rm -t bert_pyt
docker build . -f Dockerfile.pyt_hvd -t bert_pyt_hvd