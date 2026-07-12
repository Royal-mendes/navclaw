# Validation

Run:

```bash
python -m pip install pillow
python -m unittest discover -s tests -v
python -m py_compile navclaw/*.py bridge/*.py scripts/apply_verified_stop_patch.py tests/*.py
bash -n scripts/run_one_episode.sh
```

The current suite contains 34 unit/integration tests covering strict candidate/action contracts, candidate-image consistency, execution feedback memory, exact original-frontier-only filtering, verified STOP gating, CLI STOP exposure, and idempotent lower-layer FSM patching.
