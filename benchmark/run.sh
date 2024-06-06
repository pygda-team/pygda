#!/bin/sh

for i in  1 2 3
do
    echo $i
    ./run_airport.sh
    ./run_blog.sh
    ./run_citation.sh
    ./run_twitch.sh
    ./run_mag.sh
done
