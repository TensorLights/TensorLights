#!/bin/bash

vmstat $1 --one-header --unit K | while IFS= read -r line; do printf 'vmstat; %s; %s; %s\n' "$2" "$(date '+%Y-%m-%d %H:%M:%S')" "$line"; done