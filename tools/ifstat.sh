#!/bin/bash

ifstat -n -i $3 $1 | while IFS= read -r line; do printf 'ifstat; %s; %s; %s\n' "$2" "$(date '+%Y-%m-%d %H:%M:%S')" "$line"; done