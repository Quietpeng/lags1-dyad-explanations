# Public release tasks. CPU only; no restricted data is required by any target
# except the documented, non-default experiment scripts under experiments/.
PY ?= python3
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
	@rm -rf build/dist
	@mkdir -p build/dist
	@tmp=$$(mktemp); \
	GIT_INDEX_FILE=$$tmp git read-tree HEAD && \
	GIT_INDEX_FILE=$$tmp git add -A && \
	tree=$$(GIT_INDEX_FILE=$$tmp git write-tree) && \
	rm -f $$tmp && \
	git archive --format=tar.gz \
		--prefix=dyad-explanations-$(VERSION)/ \
		-o build/dist/dyad-explanations-$(VERSION).tar.gz $$tree
	@if tar -tzf build/dist/dyad-explanations-$(VERSION).tar.gz \
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
