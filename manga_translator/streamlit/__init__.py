from manga_translator.moeflow import async_detection
from .cache_async import cache_async
from pydantic import BaseModel

class TranslateTaskDef(BaseModel):
    image_file: str

@cache_async
async def start_translate_task(task_def: TranslateTaskDef):
    detection_result = await async_detection(task_def.image_file, detector_key="craft")

