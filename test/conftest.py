import pytest
from manga_translator.config import Detector

# https://docs.pytest.org/en/6.2.x/example/simple.html?highlight=addoption#pass-different-values-to-a-test-function-depending-on-command-line-options
def pytest_addoption(parser):
    parser.addoption('--translator', action='store', default=None, help='Chosen translator for test run')
    parser.addoption('--target-lang', action='store', default='ENG', help='Target language for translator test run')
    parser.addoption('--text', action='store', default=None, help='Text to be used for translation test run')
    parser.addoption('--count', action='store', type=int, default=1, help='Amount of times the test should be repeated')
    parser.addoption('--detector', action='store', default=None, help='Chosen detector for test run')
    parser.addoption('--image', default='', help='test image')
    parser.addoption('--device', default='cpu')
    parser.addoption('--detect-size', type=int, default=2048)
    parser.addoption('--text-threshold', type=float, default=0.5)
    parser.addoption('--box-threshold', type=float, default=0.7)
    parser.addoption('--unclip-ratio', type=float, default=2.3)

@pytest.fixture
def translator(request):
    return request.config.getoption('--translator')

@pytest.fixture
def tgt_lang(request):
    return request.config.getoption('--target-lang')
    
@pytest.fixture
def text(request):
    return request.config.getoption('--text')

@pytest.fixture
def count(request):
    return request.config.getoption('--count')

@pytest.fixture
def detector(request):
    return request.config.getoption('--detector')

@pytest.fixture
def image_path(request):
    return request.config.getoption('--image')

@pytest.fixture
def device(request):
    return request.config.getoption('--device')

@pytest.fixture
def detect_size(request):
    return request.config.getoption('--detect-size')

@pytest.fixture
def text_threshold(request):
    return request.config.getoption('--text-threshold')

@pytest.fixture
def box_threshold(request):
    return request.config.getoption('--box-threshold')

@pytest.fixture
def unclip_ratio(request):
    return request.config.getoption('--unclip-ratio')

@pytest.fixture
def detection_params(detector, detect_size, text_threshold, box_threshold, unclip_ratio, device):
    return {
        'detector_key': Detector(detector),
        'detect_size': detect_size,
        'text_threshold': text_threshold,
        'box_threshold': box_threshold,
        'unclip_ratio': unclip_ratio,
        'invert': False,
        'gamma_correct': False,
        'rotate': False,
        'auto_rotate': False,
        'device': device,
        'verbose': True,
    }