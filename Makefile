# Force bash instead of sh for spack compatibility
SHELL := /bin/bash
NCU := /opt/nvidia/nsight-compute/2024.3.2/ncu

.PHONY: all build clean test bench profile format check check-fix

ACTIVATE_SPACK = spack env activate cutfhe-env
SETUP_ENV = $(ACTIVATE_SPACK) && export CUDACXX=$$(which nvcc)

.ONESHELL:
all: clean
	@$(SETUP_ENV)
	@uv sync --dev --no-install-project
	@echo "Using CUDA compiler: $${CUDACXX}"
	uv pip install --editable . --force-reinstall --no-deps --no-cache

.ONESHELL:
build: clean
	@$(SETUP_ENV)
	@uv sync --dev --no-install-project
	@echo "Using CUDA compiler: $${CUDACXX}"
	uv build

clean:
	rm -rf .venv/ build/ dist/

test:
	@uv run pytest tests/ -vvv

profile:
	@$(NCU) --kernel-name mul_bootstrap_kernel --launch-count 1 -f -o profile_output --set full uv run examples/example.py

bench:
	@uv run python examples/bench_performance.py

format:
	@uvx ruff format
	@find csrc/ -type f \( -name "*.cpp" -o -name "*.h" -o -name "*.cu" -o -name "*.cuh" \) -exec uvx clang-format -i {} +

check:
	@uvx ruff check

check-fix:
	@uvx ruff check --fix
