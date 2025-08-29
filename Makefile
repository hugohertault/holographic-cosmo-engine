.PHONY: install train analyze test
install:
	python3 -m pip install -e .[dev]

train:
	hce-train --config configs/minimal.yaml --out enhanced_runs/hce_minimal

analyze:
	hce-analyze --run enhanced_runs/hce_minimal --Delta_plus 3.224744871391589

test:
	pytest -q
