import asyncio
from argparse import ArgumentParser

arg_parser = ArgumentParser()
arg_parser.add_argument("--models", default="")
arg_parser.add_argument("--continue-on-error", action="store_true")

from manga_translator.utils import ModelWrapper
from manga_translator.detection import DETECTORS
from manga_translator.ocr import OCRS
from manga_translator.inpainting import INPAINTERS


async def download(dict):
    """ """
    for key, value in dict.items():
        if issubclass(value, ModelWrapper):
            print(" -- Downloading", key)
            try:
                inst = value()
                await inst.download()
            except Exception as e:
                print("Failed to download", key, value)
                print(e)


async def main():
    parsed = arg_parser.parse_args()
    models: set[str] = set(parsed.models.split(","))
    await download(
        {k: v for k, v in DETECTORS.items() if not models or f"detector.{k}" in models}
    )
    await download(
        {k: v for k, v in OCRS.items() if not models or f"ocr.{k}" in models}
    )
    await download(
        {
            k: v
            for k, v in INPAINTERS.items()
            if (not models or f"inpaint.{k}" in models) and (k not in ["sd"])
        }
    )


if __name__ == "__main__":
    asyncio.run(main())
