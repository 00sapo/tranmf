#!/bin/env bash

# if no argument, show the description
if [ -z "$1" ]; then
	echo "This script exports all glyphs of a font to a tar file containing PNG files."
	echo "The first argument is the font file, the second argument is the directory where the tar file will be created."
	echo "The third argument is optional and is the height of the PNG files in pixels (default is 300)."
	echo "The script requires fontforge and inkscape to be installed (with flatpak or in your PATH)."
	echo "The script should be run in a directory where the user and the flatpak apps have write permissions (usually any directory in the user's home)."
	exit 0
fi

fontforge="fontforge"
if ! command -v fontforge >/dev/null 2>&1; then
	fontforge="flatpak run org.fontforge.FontForge"
	if ! flatpak list | grep org.fontforge.FontForge >/dev/null 2>&1; then
		echo >&2 "I require fontforge but it's not installed.  Aborting."
		exit 1
	fi
fi

inkscape="inkscape"
if ! command -v inkscape >/dev/null 2>&1; then
	inkscape="flatpak run org.inkscape.Inkscape"
	if ! flatpak list | grep org.inkscape.Inkscape >/dev/null 2>&1; then
		echo >&2 "I require inkscape but it's not installed.  Aborting."
		exit 1
	fi
fi

command -v tar >/dev/null 1>&1 || {
	echo "No tar command found"
	exit 1
}

# check if the first argument is a file and exists
[ -f "$1" ] || {
	echo >&2 "The first argument must be a file. Aborting."
	exit 1
}

# check if the second argument is a directory and exists
[ -d "$2" ] || {
	echo >&2 "The second argument must be a directory. Aborting."
	exit 1
}

height=300 # default height
if [ -n "$3" ]; then
	height=$3
fi

outdir=$(realpath $2)

echo "Exporting glyphs to PNG files with a height of $height pixels"

echo "Exporting $1 to SVG files"

# create temporary directory
# in order to support flatpak, avoid using /tmp and use current folder instead
tmpdir=$(mktemp -u -p $2 -t "dump_font.tmp-XXXXXX")
mkdir -p $tmpdir

cp $1 $tmpdir/
prevdir=$(pwd)
cd $tmpdir
$fontforge -lang=ff -c 'Open($1); SelectWorthOutputting(); foreach Export("%u-%e-%n.svg"); endloop;' $(realpath $1) >/dev/null 2>&1

echo "Converting SVG files to PNG into a tar file: $2/$(basename $1 .ttf).tar"

# Function to update a progress bar
update_progress_bar() {
	local current=$1
	local total=$2
	local bar_length=$(($(tput cols) - 20))
	local progress=$((current * bar_length / total))
	local remaining=$((bar_length - progress))
	local progress_bar=$(command printf '#%.0s' $(seq 1 $progress))
	local empty_bar=$(command printf '%.0s' $(seq 1 $remaining))

	# Clear the line and print the progress bar
	echo -ne "\rProgress: [${progress_bar}${empty_bar}] ${current}/${total}"
}

files=$(ls *.svg)
# count the number of $files
nfiles=$(echo $files | wc -w)
# count width of terminal
progress=0
for file in $files; do
	# print the bar progress
	update_progress_bar $((++progress)) $nfiles

	# run inkscape
	$inkscape $(realpath $file) --export-type=png --export-filename=$(basename $file .svg).png --export-area-drawing -h $height >/dev/null 2>&1
done

# put all akk the $tmpdir/*.png files into a tar file
tar -cf $outdir/$(basename $1 .ttf).tar *.png

# remove the temporary directory
# rm -rf $tmpdir
cd $prevdir
