#!/bin/bash
set -e
python main.py start &
python token_server.py
