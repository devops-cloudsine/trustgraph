# TrustGraph Development Workflow

This repository now ships with a **dev container** workflow so you can test
changes to `trustgraph-flow` and `trustgraph-base` without rebuilding images
after every edit.

## One-time setup

1. Build the development image:

   ```bash
   cd /home/ubuntu/Github/trustgraph
   make container-trustgraph-flow-dev
   ```

   This uses `Containerfile.dev` to install both packages in editable mode.

2. Ensure Docker Compose picks up the override file by restarting the stack
   from `/home/ubuntu`:

   ```bash
   docker-compose down
   docker-compose up -d
   ```

   The override switches every `trustgraph-flow` service to the `:dev` image
   and bind-mounts your source tree into the container.

## Iteration loop

1. Edit the code under
   `/home/ubuntu/Github/trustgraph/trustgraph-flow/trustgraph` or
   `/home/ubuntu/Github/trustgraph/trustgraph-base/trustgraph`.
2. Restart only the affected services, e.g.:

   ```bash
   docker-compose restart document-retrieval
   ```

3. Test immediately (via `curl`, Postman, test scripts, etc.).

Editable installs mean the container imports your live workspace, so no image
rebuild is necessary unless you change dependencies.

## When to rebuild the dev image

Run `make container-trustgraph-flow-dev` again if you:

- Add or upgrade Python dependencies (`pyproject.toml`).
- Modify `Containerfile.dev`.
- Need a clean virtual environment.

For all other code changes, simply restart the relevant services.

## Important: Namespace Package Structure

`trustgraph` is a **PEP 420 namespace package** that combines modules from both `trustgraph-base` and `trustgraph-flow`. For this to work correctly:

- **DO NOT** have `trustgraph/__init__.py` at the root level in either package
- **DO** have `__init__.py` files in subdirectories (like `trustgraph/config/__init__.py`)

If you add a root-level `trustgraph/__init__.py`, Python will treat it as a regular package and the two packages won't merge properly, causing `ModuleNotFoundError` for modules in the other package.

