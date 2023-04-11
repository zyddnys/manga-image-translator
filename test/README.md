## Tests

### Installation

```bash
pip install pytest pytest-asyncio
```

### Usage:

Run all tests
```bash
pytest test/
```

Run specific translator test
```bash
pytest test/test_translation_manual.py --translator sugoi --target-lang ENG
```

To disable stdout capture add `-s --log-cli-level=DEBUG`
