import asyncio

from manga_translator.utils import ModelWrapper
from manga_translator.detection import DETECTORS
from manga_translator.ocr import OCRS
from manga_translator.inpainting import INPAINTERS

async def download(dict):
  for key, value in dict.items():
    if issubclass(value, ModelWrapper):
      print(' -- Downloading', key)
      inst = value()
      await inst.download()

async def main():
  await download(DETECTORS)
  await download(OCRS)
  await download({
    k: v for k, v in INPAINTERS.items() 
      if k not in ['sd']
  })

if __name__ == '__main__':
  asyncio.run(main())
