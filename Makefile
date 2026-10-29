# Public release tasks. CPU only; no restricted data is required by any target
# except the documented, non-default experiment scripts under experiments/.
# Prefer a local virtualenv when present; a fresh checkout still uses python3
# and the README documents `python3 -m venv .venv` plus the pinned install.
PY ?= $(if $(wildcard .venv/bin/python),.venv/bin/python,python3)
VERSION ?= 0.1.1

export PYTHONDONTWRITEBYTECODE := 1
export MPLBACKEND := Agg

.PHONY: doctor test verify notebooks stability-check worldline-check dist clean

doctor:
	$(PY) scripts/doctor.py

test:
	$(PY) -m unittest discover -s tests -v

verify:
	$(PY) scripts/verify_release.py

notebooks:
	$(PY) scripts/execute_notebooks.py

# These targets verify only the aggregates shipped in results/. The ICEWS
# event panel, experiment configuration and row-level predictions are not
# redistributed, so the audit itself cannot be re-run from this tree.
# stability-check pins the 50-refit scope metadata and stability/SHA256SUMS;
# it never runs experiments/ and never writes results.
stability-check:
	$(PY) scripts/check_aggregates.py stability

worldline-check:
	$(PY) scripts/check_aggregates.py worldline

dist:
	@set -eu; \
	if [ -n "$$(git status --porcelain --untracked-files=no)" ]; then \
		echo "ERROR: tracked files are modified or staged; commit before make dist"; \
		git status --short --untracked-files=no; \
		exit 1; \
	fi; \
	rm -rf build/dist; \
	mkdir -p build/dist; \
	git archive --format=tar.gz \
		--prefix=dyad-explanations-$(VERSION)/ \
		-o build/dist/dyad-explanations-$(VERSION).tar.gz HEAD; \
	archive=build/dist/dyad-explanations-$(VERSION).tar.gz; \
	git ls-tree -r --name-only HEAD | LC_ALL=C sort > build/dist/head-files.txt; \
	tar -tzf "$$archive" \
		| sed -e 's|^dyad-explanations-$(VERSION)/||' \
		| grep -v '/$$' | grep -v '^$$' \
		| LC_ALL=C sort > build/dist/archive-files.txt; \
	if ! diff -u build/dist/head-files.txt build/dist/archive-files.txt; then \
		echo "ERROR: archive file list differs from HEAD"; \
		exit 1; \
	fi; \
	if tar -tzf "$$archive" \
		| grep -E '(^|/)(paper|docs|configs|data|preds|__pycache__|\.git)(/|$$)'; then \
		echo "ERROR: forbidden path in archive"; exit 1; \
	fi
	@rm -rf build/dist/verify
	@mkdir -p build/dist/verify
	@tar -xzf build/dist/dyad-explanations-$(VERSION).tar.gz \
		-C build/dist/verify --strip-components=1
	$(PY) scripts/verify_release.py --root build/dist/verify
	@echo "archive: build/dist/dyad-explanations-$(VERSION).tar.gz"

clean:
	find . -type d \( -name __pycache__ -o -name .ipynb_checkpoints \
		-o -name .pytest_cache -o -name .mypy_cache -o -name .ruff_cache \) \
		-prune -exec rm -rf {} +
	rm -rf build dist *.egg-info .coverage htmlcov
