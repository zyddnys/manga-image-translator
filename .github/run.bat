@echo off

pushd "%~dp0"
git pull --quiet
python -m manga_translator %*
popd
