# Validation

Run:

```bash
python -m pip install pillow
python -m unittest discover -s tests -v
python -m py_compile navclaw/*.py bridge/*.py tests/*.py
bash -n scripts/run_one_episode.sh
```

The current suite contains 24 unit/integration tests covering strict candidate/action contracts, candidate-image consistency, execution feedback memory, and exact original-frontier-only filtering.
