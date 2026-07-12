# Validation

Run:

```bash
python -m pip install pillow
python -m unittest discover -s tests -v
python -m py_compile navclaw/*.py bridge/selector_client.py tests/*.py
```

The current contract and feedback suite contains 19 unit/integration tests.
