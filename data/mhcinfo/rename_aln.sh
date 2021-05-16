#!/bin/sh

for fn in $(ls *.aln)
do
	species=$(echo $fn | cut -d'-' -f1)
#	echo "species: $species"
	new_fn=$(echo $fn |  sed -e "s/$species-\(.*\)-protein.aln/\1.aln/")
#	echo "new_fn: $new_fn"
	mv $fn $new_fn
done
