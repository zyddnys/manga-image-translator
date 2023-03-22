import asyncio

from manga_translator.utils import ModelWrapper
from manga_translator.detection import DETECTORS
from manga_translator.translators import TRANSLATORS
from manga_translator.ocr import OCRS
from manga_translator.inpainting import INPAINTERS
from manga_translator.upscaling import UPSCALERS

async def download(dict):
  for key, value in dict.items():
    if issubclass(value, ModelWrapper):
      print(' -- Downloading', key)
      inst = value()
      await inst.download()

async def main():
  await download({
    k: v for k, v in TRANSLATORS.items() 
      if k not in ['offline', 'offline_big', 'nllb_big', 'jparacrawl']
  })
  await download(DETECTORS)
  await download(OCRS)
  await download({
    k: v for k, v in INPAINTERS.items() 
      if k not in ['sd']
  })
  await download(UPSCALERS)

if __name__ == '__main__':
  asyncio.run(main())
