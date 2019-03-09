#!/bin/bash

now=$(date +%s)
if [ $(( $4 - $now )) -ge 0 ]; then
	echo "Going to wait until < $(date -d @$4) > before starting tcpdump"
	sleep $(( $4 - $now ))
fi
timeout $5 tcpdump -i $3 -n | while IFS= read -r line; do printf 'tcpdump; %s; %s %s\n' "$2" "$(date '+%Y-%m-%d')" "$line"; done