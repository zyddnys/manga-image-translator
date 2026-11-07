if [[ -d "$1" ]]; then
    echo "Running live eng editor..."
else
    echo "Please provide a valid directory. e.g. ./run_live_editor.sh path-to-image-directory"
    exit 0
fi
OVERWRITE=--overwrite
python -m manga_translator local  --config-file ./editor-config.json --font-path fonts/anime_ace.ttf $OVERWRITE --multi-gpu --use-gpu --live-editor -i $1 
