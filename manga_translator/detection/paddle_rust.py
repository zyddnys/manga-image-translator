from manga_translator.detection.common_rust import RustDetector, get_session


class PaddleDetector(RustDetector):
    def __init__(self):
        super().__init__(get_session().paddle_detector())
