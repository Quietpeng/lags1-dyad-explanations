PY ?= python
PANDOC ?= pandoc

.PHONY: test smoke worldline paper clean repro

test:
	$(PY) -m unittest discover -s tests -v

smoke: test

worldline:
	$(PY) experiments/worldline_preview.py --threads 1

paper:
	$(PY) paper/build_paper.py

# Full P1 reproduction requires the restricted data/icews snapshot.
# It is not run by default and will fail on a clean clone without data access.
repro:
	$(PY) -m framework.run_pilot --config configs/framework_pilot.json --stage all

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .pytest_cache
	rm -f paper/*.aux paper/*.log paper/*.out paper/*.toc
	rm -f paper/figures/*.aux paper/figures/*.log
