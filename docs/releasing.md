# Releasing cadjoint to PyPI

cadjoint is published to PyPI as `cadjoint` by
[`.github/workflows/release.yml`](../.github/workflows/release.yml) using
[Trusted Publishing](https://docs.pypi.org/trusted-publishers/). No API token
exists: PyPI trusts a short-lived OIDC token that GitHub mints for that
workflow, in that repository, running in a named environment.

## Decisions

**Name.** `cadjoint` has no files on PyPI or TestPyPI (`/pypi/cadjoint/json`
and `/simple/cadjoint/` both 404, checked 2026-10-16). A project that was
registered but never released also 404s there, so the name is settled only
when the pending publisher below is accepted. If PyPI refuses it, the
fallback is `cadjoint-jax` (also free). The import name stays `cadjoint`
either way.

**Version.** The single source is `version` in `pyproject.toml`. The package
reads it back at runtime (`cadjoint.__version__`, via `importlib.metadata`),
and the release workflow refuses a tag that disagrees with it. The scheme is
semantic versioning under the pre-1.0 rule: until 1.0, a minor bump
(`0.1` → `0.2`) may break the API and a patch bump (`0.1.0` → `0.1.1`) may
not. The README and the PyPI page say the API is unstable and that
dependents should pin an exact version. Pre-releases use PEP 440 suffixes
(`0.2.0rc1`, tag `v0.2.0rc1`). Changes are recorded in
[`CHANGELOG.md`](../CHANGELOG.md).

**What the wheel carries.** The `cadjoint` package and nothing else. That
includes the committed playground build (`cadjoint/viewer/static`), the
TypeScript wire schema, and each Tesseract package's
`tesseract_config.yaml` and requirements or environment file. The plugin
registry reads a built-in's version from its config, which an earlier wheel
dropped. `research/`, `scenes/`, `examples/`, `frontend/`, `docs/` and
`tests/` stay out of both the wheel and the sdist. The wheel is about 1.2 MB,
most of it the playground's JavaScript bundle.

**Licence metadata.** `license = "Apache-2.0"` (an SPDX expression, PEP 639)
and `license-files = ["LICENSE"]`. This needs setuptools ≥ 77, which the
build-system pins.

**README on PyPI.** PyPI renders the same `README.md` but cannot resolve
repository-relative paths. Images point at
`raw.githubusercontent.com/andrinr/cadjoint/main/docs/assets/...`, and links
point at the docs site or at `github.com/.../blob/main/...`. They follow
`main`, so an old release's page shows today's clips. Keep new README links
absolute; `twine check --strict` in the workflow catches markup errors, not
relative links.

**Dependencies.** Every requirement in every extra is a plain PyPI name.
PyPI rejects direct URL and git references, and none remain. The `[tool.uv]`
settings (`build-constraint-dependencies` and the `dependency-metadata`
entries for petsc and petsc4py) only steer uv's own locking and source builds.
They never reach the published metadata and need no change.

- `fem` stays an extra. petsc4py publishes only an sdist, so
  `pip install "cadjoint[fem]"` compiles PETSc and needs a C compiler,
  gfortran and BLAS/LAPACK headers. The extra still states correctly what
  `cadjoint.fem`'s jax-fem path imports. A user with a conda-forge petsc4py
  already installed has it satisfied without a build, and moving the list
  into prose would make the extra silently incomplete instead. The README's
  install section names the toolchain, and the prebuilt GHCR solver images
  avoid it.
- `gmsh` stays an extra under its GPL caveat. Listing a GPL package as an
  optional dependency does not change cadjoint's Apache-2.0 licence, and
  nothing imports gmsh unless the extra is used.
- `cuda`, `tesseract`, `viewer`, `editor` and `stepcheck` resolve from PyPI
  as they are. `dev` and `docs` ship as extras too, which is harmless.
- `cadjoint-viewer` is installed with the core package but needs the
  `viewer` extra (pydantic). Without it, the script prints the install command
  instead of a traceback.
- A Tesseract package inside an installed wheel cannot be built with
  `tesseract build`, because its requirements install cadjoint from the
  repository root (`../../../..`). Build Tesseracts from a checkout, or use the
  GHCR images.

## The workflow

| Trigger | Version published | Where |
| --- | --- | --- |
| Actions → release → *Run workflow* | `X.Y.Z.devN` (N = run number) | TestPyPI only (dry run) |
| push of tag `vX.Y.Z` | `X.Y.Z` | TestPyPI, then PyPI |

The jobs, in order:

1. **build** checks the tag against `pyproject.toml` (or rewrites the version
   to `.devN` for a dry run), then runs `uv build` (the sdist, then the wheel
   built from that sdist) and `twine check --strict`.
2. **smoke** installs the wheel alone into a fresh venv on Linux with Python
   3.11, 3.12 and 3.13, and on macOS with 3.12. It then runs
   [`.github/scripts/smoke_installed.py`](../.github/scripts/smoke_installed.py)
   from outside the checkout. The script checks the import location and
   version, runs the README example with a `jax.grad` through it, and checks
   the package data. It also checks that `cadjoint-viewer --help` works with
   `[viewer]` installed.
3. **testpypi** publishes to TestPyPI (environment `testpypi`).
4. **testpypi-install** downloads the wheel back from TestPyPI by version,
   checks its sha256 against the build, installs it with dependencies from
   PyPI, and runs the smoke script again.
5. **pypi** publishes to PyPI (environment `pypi`). It runs on tag pushes only.

## One-time setup (Andrin)

Nothing below can be done from the repository.

1. **PyPI pending publisher.** Signed in at <https://pypi.org>, go to *Your
   account → Publishing → Add a new pending publisher → GitHub*:
   - PyPI project name: `cadjoint`
   - Owner: `andrinr`
   - Repository name: `cadjoint`
   - Workflow name: `release.yml`
   - Environment name: `pypi`

   The pending publisher does not reserve the name. The project is created,
   and owned by your account, on the first successful upload.
2. **TestPyPI pending publisher.** TestPyPI is a separate service with its
   own account. At <https://test.pypi.org>, make the same entry with
   environment name `testpypi`.
3. **GitHub environments.** In *andrinr/cadjoint → Settings → Environments*,
   create `testpypi` and `pypi`. On `pypi`, add yourself as a required
   reviewer and restrict deployments to tags matching `v*`. Each real release
   then waits for one click, and a pushed branch can never publish.
4. **Optional:** in *Settings → Rules*, protect tags matching `v*` so only
   you can create them.

## First release

1. Merge the packaging branch into `main`.
2. Dry run: *Actions → release → Run workflow* on `main`. This publishes
   `0.1.0.devN` to TestPyPI, installs it back, and smoke-tests it. Check
   <https://test.pypi.org/project/cadjoint/> for the README's rendering:
   images, tables, and the `[!WARNING]` block, which PyPI may render as a
   plain quote.
3. Tag and push:

   ```bash
   git tag -a v0.1.0 -m "cadjoint 0.1.0"
   git push origin v0.1.0
   ```

4. Approve the `pypi` deployment when the run asks. Then check
   `pip install cadjoint==0.1.0` in a fresh venv.
5. Optionally, create a GitHub release from the tag with the `0.1.0` section
   of the changelog.

## Every later release

1. Move the `Unreleased` entries in `CHANGELOG.md` under the new version and
   update the compare links at the bottom.
2. Bump `version` in `pyproject.toml` and run `uv lock`. The lock records
   cadjoint's own version, and `uv sync --locked` in `install.yml` fails on a
   stale lock.
3. Merge that to `main`, then tag `vX.Y.Z` on the merge commit and push the
   tag.

A run that fails after the TestPyPI upload can be re-run: the TestPyPI step
skips files that are already there. A version that reached PyPI can never
be uploaded again, even after deletion, so a broken release is fixed with a
new patch version.

## Downstream: the text-to-cad skill

The cadjoint skill in earthtojake/text-to-cad#639 pins
`cadjoint @ git+https://github.com/andrinr/cadjoint@d2aa5e74…` in
`skills/cadjoint/requirements.txt` and `requirements-dev.txt`. That commit is
an ancestor of `main`, and the skill imports only core modules
(`construction`, `constraints`, `geometry`, `sdf`), so `0.1.0` covers it.
Once `0.1.0` is on PyPI, both lines become:

```text
cadjoint==0.1.0
```

That removes the git dependency the review objected to. Installs need no git
and no source checkout. pip and uv can cache and hash-pin the wheel, and the
requirement can be resolved alongside other packages, which a direct
reference blocks once anything published depends on it. The exact pin keeps
the reason the commit pin existed: the API is unstable, so the skill moves to
a new cadjoint version deliberately. The `# cadjoint is not on PyPI` comment
in the skill's requirements file goes. jax remains the heavy part of the
install, and that does not change.
