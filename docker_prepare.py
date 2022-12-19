import asyncio

from utils import ModelWrapper
from translators import TRANSLATORS
from ocr import OCRS
from inpainting import INPAINTERS
from upscaling import UPSCALERS

async def download(dict):
  for key, value in dict.items():
    if issubclass(value, ModelWrapper):
      print(' -- Downloading', key)
      inst = value()
      await inst.download()

async def main():
  await download({
    k: v for k, v in TRANSLATORS.items() 
      if k not in ['offline_big', 'nllb_big', 'sugoi_small', 'sugoi_big']
  })
  await download(OCRS)
  await download(INPAINTERS)
  await download(UPSCALERS)

if __name__ == '__main__':
  asyncio.run(main())
