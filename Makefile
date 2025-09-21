.PHONY: build up down run

build:
\tdocker compose build

up:
\tdocker compose up -d

down:
\tdocker compose down

run:
\t@echo "Usage: make run SCRIPT=scripts/deep_build.py"
\tdocker compose run --rm bot python $(SCRIPT)
