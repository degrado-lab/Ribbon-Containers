---
name: build-ribbon-container
description: >-
  Procedure and runbook for building, packaging, and validating Docker/Apptainer containers 
  in Ribbon-Containers on the Slurm lab host. Use whenever creating a new container or modifying existing build definitions.
---


# Building a container in Ribbon-Containers

Process for adding a new software container to
`~/workspace/Ribbon-Containers` on the lab host and validating it well
enough to trust. Written from one complete run (ESMFold2 26.9.5, a
14.6 GB GPU image with ~14 GB of model weights baked in).
This container repo is for many types of software, so adapt to the specific use-case. Follow the phases in order; the ordering is what keeps you from discovering a blocker after a 40-minute build.

**Host:** Local Server: (Slurm, single node `LocalQ`). Docker
28.1.1 and Apptainer 1.4.3, daemon access without sudo.

---

## Phase 0 — Read the repo before you write anything

This repo has conventions and they are not guessable. Read these, in
this order, before proposing a design:

```bash
R=~/workspace/Ribbon-Containers
cat $R/README.md                    # naming + version rules
cat $R/containers/README.md          # build best practices
cat $R/templates/Dockerfile $R/templates/definition.def
cat $R/build.sh $R/build-docker.sh $R/build-app.sh
cat $R/.github/workflows/build-container-local-app-only.yml
ls -1 $R/containers                  # pick your precedents
```

### The conventions you must honour

- Layout is `containers/<Software>/<Version>/`, holding `Dockerfile`,
  `definition.def`, `README.md`, optionally example data.
- **One primary piece of software per container.** Resist bundling your
  own analysis code into the image — it belongs outside.
- Directory uses proper capitalization; the *uploaded* name is all
  lowercase (`ESMFold2` → `esmfold2`).
- Version matches the primary software's version. If the software has no
  version, use the release **date** in `YY.M.D` (e.g. `26.9.5`). If you
  re-release the container without a software bump, append a decimal
  (`5.0.3` → `5.0.3.1`).
- Tags: Apptainer/ORAS gets `<version>`; Docker gets `docker-<version>`.
- Updating a container means a **new version directory**, never editing
  an existing one.
- README format is fixed: `### Usage` (Docker + Apptainer command lines),
  `### Implementation Notes`, `### To Do`.

### Pick two precedents, not zero

Find the container in `containers/` that is closest to yours and read
its `Dockerfile`, `definition.def` and `README.md` in full. Also read
the closest container that **failed** at what you're attempting — the
notes on why are worth more than the successes.

For the ESMFold2 build those were `Boltz/2.2.0` (the working pattern for
baking model weights at build time via a `download_weights.py`) and
`ESM3/3.2.0`, whose README records that baked weights were abandoned
because the model is gated and its loader wants to *write* into the
weights directory at runtime — impossible on Apptainer's read-only
filesystem. That note set the two things the new build had to prove:
weights ungated, and nothing written at runtime. Generally all relevant 
files and weights must be inside the container - flag the user's attention 
if weights are gated.

---

## Phase 1 — Host reconnaissance, before planning

Verify rather than assume. Each of these has bitten a build:

1. **Is the base image actually published?** The repo contains a
   directory for a base tag that was never pushed. Check DockerHub for
   the tag you intend to use, and fall back to the tag the working
   precedents actually pull (`nicholasfreitas/cuda_micromamba_base:cuda12.2.0-base-ubuntu22.04-micromamba`).
2. **Disk on the root filesystem**, which is where `overlay2` lives —
   not `/scratch`. A weight-baking build holds the downloaded and the
   converted copies simultaneously.
3. **Are the weights gated?** Fetch the manifest anonymously and check
   for a redirect rather than a 401. If gated, follow the ESM3
   precedent and do not bake.
4. **Does the loader accept a local path?** If the only way to point the
   software at weights is a Hub repo id, you will have to patch configs.
   Find the knob before you plan.
5. **Toolchain versions and GPU**: `docker version`, `apptainer
   version`, `nvidia-smi`, and the driver's CUDA version.

Record the resolved answers in a small `build_env.json` in the version
directory — base tag, tool versions, free disk. It costs nothing and
makes the build auditable later.

---

## Phase 2 — Ask the user the decisions you cannot make

Two classes of decision are theirs, and asking takes one round-trip:

- **The version string.** Whether the software's own version or a
  `YY.M.D` date applies is a judgement about the upstream project.
- **Any change to numerics.** For ESMFold2, storing the 24.4 GB fp32
  checkpoint pre-cast to bf16 cut the image nearly in half and was
  arguably free — the loader's default already casts to bf16 at load.
  But "arguably free" is a claim about their science, not yours. Ask,
  state what is bit-identical and what is not, and record the answer in
  the provenance file.

Do not ask about anything you can check on the host yourself.

---

## Phase 3 — Plan, with the validation step named up front

The eight steps that worked, in order:

1. Verify base image, stage the version directory, write `build_env.json`
2. Write the Dockerfile with pinned installs
3. Write the weight-bake script
4. Write the runtime helper and `definition.def`
5. Write the README in the repo's format
6. Build the Docker image
7. Build the SIF and **validate stand-alone**
8. Deliver and record

Step 7 is the one that earns the work. Write it into the plan as "fold a
real target with the network off and score the output against a known
reference", not as "test the container" — otherwise it degrades into
running `--help` and calling it done.

---

## Phase 4 — Write the version directory

### Dockerfile: two stages, following the template

Stage 1 (builder) on the CUDA + micromamba base:

```dockerfile
RUN micromamba create -y -n app -c conda-forge python=3.12 pip
RUN micromamba run -n app pip install --no-cache-dir <pinned specs>
RUN micromamba clean -a -y
RUN rm -rf /opt/conda/pkgs /root/.cache/pip
RUN find /opt/conda/envs/app -name "*.a" -delete || true
```

Stage 2 (runtime) on the same base, copying only what's needed:

```dockerfile
COPY --from=builder /opt/conda/envs/app /opt/conda/envs/app
COPY --from=builder /app/models /app/models
SHELL ["/bin/bash", "-c"]
RUN echo 'eval "$(micromamba shell hook --shell bash)"' >  /etc/profile.d/mamba.sh && \
    echo 'micromamba activate app'                      >> /etc/profile.d/mamba.sh
ENV PATH=/opt/conda/envs/app/bin:$PATH
WORKDIR /workspace
```

Rules that matter:

- **Pin everything**, including git installs, to exact commits. A
  container whose contents drift is not a container.
- **Deviations from the template get documented.** ESMFold2 needed
  python 3.12, not the template's 3.11, because the upstream fork
  requires `>=3.12,<3.13`. Note it in the README rather than leaving the
  next reader to wonder.
- **Set offline flags in the image**, not just in the definition:
  `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`. If a code path can reach
  the network it eventually will.
- **Ship a PATH shim for your entry point.** Do not rely on the
  runscript passing arguments through — see the failure catalogue.

### Weight baking

Model on `Boltz/2.2.0`'s `download_weights.py`. By default, containers should be "battery-included" with weights baked in. Specifics that mattered:

- **Convert without instantiating the model.** Stream-convert safetensors
  shards with the `safetensors` API — a few GB of RAM and no GPU, rather
  than loading a 6B model.
- **Preserve every non-tensor file in the snapshot.** The one that broke
  the build was `ccd.pkl`, a chemical-component dictionary shipped
  alongside the weights and loaded by a completely different code path
  from the model.
- **Rewrite the index.** `model.safetensors.index.json` carries
  `total_size`; a dtype conversion invalidates it.
- **Patch Hub repo ids to in-image paths** in the configs (e.g.
  `esmc_id: biohub/ESMC-6B` → `/app/models/ESMC-6B`).
- **Delete the originals in the same `RUN` layer**, or they persist in
  the image regardless of later removal.
- **Emit `/app/models/WEIGHTS.json`** recording source repo, commit sha,
  original dtype, and any conversion. This is what makes `%test`
  possible and the image auditable from inside.

### definition.def

```
Bootstrap: docker
From: <dockeruser>/<name>:docker-<version>

%labels
    Maintainer <name>
    Description <one line>

%environment
    export MAMBA_ROOT_PREFIX=/opt/conda
    export PATH=/opt/conda/envs/app/bin:$PATH
    export <SOFTWARE>_MODELS=/app/models
    export HF_HUB_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
    export PYTHONPATH=/app
    # Every cache must leave the read-only image. Apptainer mounts /tmp
    # writable even under --containall.
    export HF_HOME=/tmp/hf
    export XDG_CACHE_HOME=/tmp/cache
    export TRITON_CACHE_DIR=/tmp/triton
    export TORCHINDUCTOR_CACHE_DIR=/tmp/inductor
    export MPLCONFIGDIR=/tmp/mpl
    if [ -n "$PS1" ]; then source /etc/profile.d/mamba.sh; fi

%runscript
    #!/bin/bash
    if [ $# -eq 0 ]; then
        bash --login -i
    else
        micromamba run -n app "$@"
    fi

%test
    # No GPU needed. Assert the image is self-contained.
    /opt/conda/envs/app/bin/python -c "
import json, os
m = json.load(open('/app/models/WEIGHTS.json'))['models']
assert set(m) == {...}, sorted(m)
...
"
```

Keep `Bootstrap: docker` in the **shipped** file — the CI workflow
expects it and it resolves once the tag is public. You will override it
locally (Phase 6).

**The cache redirection block is load-bearing.** Torch, Triton, HF and
matplotlib all want writable cache directories, and the SIF is
read-only. Every one of those must be pointed at `/tmp`.

**Put a regression guard in `%test` for every stand-alone defect you
find.** After discovering that one library ignored the models
environment variable and needed its own override, the assertion added
was:

```python
ccd = os.environ.get('ESMCFOLD_CCD_PATH', '')
assert ccd and os.path.exists(ccd), ('ESMCFOLD_CCD_PATH', ccd)
```

`%test` runs during `apptainer build`, so a regression fails the build
rather than surfacing months later on someone else's machine.

### Test inputs

Stage small reference inputs in `test/` under the version directory, and
**fetch reference sequences from the source database rather than typing
them** — for ESMFold2, two JSON specs built from RCSB entries: one small
apo protein for a smoke test, one protein–ligand complex for the case
that actually exercises the interesting code path.

---

## Phase 5 — Build the Docker image

Dispatch as a remote job; no GPU needed. Capture the full log, and grep
the log for the things you care about rather than reading it all:

```bash
docker build --progress=plain -t <user>/<name>:docker-<ver> . > build_docker.log 2>&1
RC=$?
grep -E '<your version-pin gate>' build_docker.log | tail -2
grep -E '=== |BAKED|GB ->' build_docker.log | tail -50
docker images --format '{{.Repository}}:{{.Tag}} {{.Size}}' | grep <name>
df -h /var/lib/docker | tail -1
gzip -9f build_docker.log
exit $RC
```

Two habits:

- **`set +e` around the build and propagate `$RC` explicitly**, so the
  log is harvested on failure. A job that dies before its log transfers
  costs you the whole build.
- **Print a build-time gate** for anything you've pinned — have the
  Dockerfile echo the installed version and grep for it. This catches a
  silently-resolved-different dependency without a GPU or a run.

Expect the builder stage to cache. A re-build after a runtime-stage edit
was 333 s, not a re-bake.

---

## Phase 6 — Build the SIF from the local daemon

Nothing has been pushed yet, so `Bootstrap: docker` cannot resolve.
Build from the local Docker daemon by rewriting only the bootstrap line,
and leave the shipped file untouched:

```bash
D=$HOME/workspace/Ribbon-Containers/containers/<Software>/<Version>; cd $D
export APPTAINER_TMPDIR=/scratch/freitas/.apptainer_tmp
export APPTAINER_CACHEDIR=/scratch/freitas/.apptainer_cache
mkdir -p $APPTAINER_TMPDIR $APPTAINER_CACHEDIR
sed 's|^Bootstrap: docker$|Bootstrap: docker-daemon|' definition.def > definition.local.def
apptainer build --force --mksquashfs-args '-mem 512M -processors 2' \
    <name>_<version>.sif definition.local.def
rm -f definition.local.def
apptainer test <name>_<version>.sif
apptainer inspect <name>_<version>.sif
```

**`APPTAINER_TMPDIR` must be on `/scratch`.** The default is under `/tmp`
and a multi-GB image will fill it.

**`--mksquashfs-args '-mem 512M -processors 2'` is mandatory on this
host.** See the failure catalogue — this one cost a full build cycle.

---

## Phase 7 — Validate stand-alone. This is the step that matters.

Run the real workload inside the SIF with the host removed:

```bash
apptainer run --nv --containall --net --network=none <name>_<version>.sif \
    <entrypoint> --input test/<case>.json --out out/
```

- `--containall` — no host `$HOME`, no host `/tmp`, no bind mounts
- `--net --network=none` — no route anywhere

Then **score the output against a known reference**, not just check it
exists. For ESMFold2 that meant CA RMSD and TM-score against the
experimental structures, plus ligand heavy-atom RMSD for the complex:
1UBQ at 0.94 Å median / TM 0.951, 1STP at 0.38 Å / TM 0.991 with biotin
at 0.48 Å. Record fold time and peak VRAM in
`container_validation.json`.

### Why the isolation flags are the whole point

On the bare host this container would have passed. The user's home HF
cache already held `ccd.pkl`, so a library that silently
`hf_hub_download`s that file would have found it locally — and the
container would have looked self-contained on the only machine anyone
would have tested it on, then failed on every other one. The defect only
surfaced with the network off and `$HOME` unmounted.

**If the image claims to be self-contained, the test must remove
everything it could be leaning on.** Anything less tests the host.

Also record what the validation reveals about limits. Peak VRAM was
13.98 GB at 76 tokens and 15.75 GB at 175 on a 16 GB card, ~12.7 GB of
it resident model — so activations set a ceiling near 200 tokens for the
full model on that GPU. That belongs in the README, because it decides
whether a user reaches for the smaller variant.

---

## Phase 8 — Deliver, record, and leave the push to the user

- Sync the version directory into the repo with `README.md`,
  `build_env.json`, `container_validation.json`.
- Save the build log and validation record as artifacts.
- Append durable host facts to the compute notes so the next session
  finds the SIF instead of rebuilding it.
- **Do not push.** The wrappers use the user's DockerHub secrets. Report
  the command and stop:

  ```bash
  cd ~/workspace/Ribbon-Containers && ./build.sh <Software> <Version>
  ```

  `build.sh` runs both stages via `act`; `build-docker.sh` and
  `build-app.sh` run one each. All three take
  `<container-name> <container-version>`, read
  `../secrets/.secrets`, and map `ubuntu-latest=-self-hosted`.

- **Disclose any drift between the SIF on disk and the source.** The
  built image predated a corrected `%labels Description` string; the
  wrapper run refreshes it, but the user needed telling.

---

## Failure catalogue

Every one of these was hit in the ESMFold2 build.

**`mksquashfs` OOM-killed at the end of a successful build.** The log
ends `while creating squashfs: mksquashfs command failed: signal:
killed` after `%test` has already passed, and the Slurm output says
`Detected 1 oom_kill event`. Cause: `mksquashfs` sizes its cache at ~25%
of the box's physical RAM (62 GB here), while the job's cgroup is
`DefMemPerCPU=2000` MB. The job wrapper emits **no `#SBATCH` memory
directives**, and passing `scheduler=` to the submit call is accepted but
adds none — verified by reading the generated `job.sh`. So the
allocation cannot be raised from the submitting side; cap the compressor
instead: `--mksquashfs-args '-mem 512M -processors 2'`.

**Node limits.** `LocalQ`: 10 CPUs, 32000 MB, `gpu:1`.
`DefMemPerCPU=2000`, `MaxMemPerCPU=8000`. A memory request above the
node's capacity is rejected at submit, and the rejection lands in a
default-named file in the submitting directory rather than the
configured output path — so the job looks like it failed silently.

**A library ignoring your models environment variable.** One import-time
loader bypassed `<SOFTWARE>_MODELS` entirely and had its own, differently
spelled override (`ESMCFOLD_CCD_PATH`, not `ESMFOLD2_...`). Baking the
file was not enough; the path has to be set in the Dockerfile, in
`%environment`, *and* passed by the entry script. Then guard it in
`%test`.

**`micromamba run -n app "$@"` swallowing your flags.** The activation
wrapper ends in a bare `exec "$@"`, so `--input` is parsed as an option
to `exec` and the run dies. Fix: install a shim on `PATH` inside the
image that execs the interpreter directly.

```dockerfile
RUN printf '#!/bin/bash\nexec /opt/conda/envs/app/bin/python /app/<tool>.py "$@"\n' \
      > /opt/conda/envs/app/bin/<tool> && chmod +x /opt/conda/envs/app/bin/<tool>
```

**Read-only filesystem cache writes.** Any of Torch, Triton,
TorchInductor, HF or matplotlib will try to write a cache and fail under
Apptainer. Redirect all of them to `/tmp` in `%environment`.

**A base image directory that was never published.** Check the registry,
not the repo.

---

## Checklist

- [ ] Repo READMEs, templates, build wrappers and workflow read
- [ ] Closest working precedent and closest *failed* precedent read
- [ ] Base image tag confirmed published
- [ ] Weights confirmed ungated; loader confirmed to accept a local path
- [ ] Root-filesystem disk headroom checked
- [ ] Version string and any numerics change confirmed with the user
- [ ] Version directory named per convention; no existing version edited
- [ ] All installs pinned to exact versions/commits
- [ ] Template deviations documented in the README
- [ ] Provenance file baked into the image
- [ ] `%test` asserts self-containment, with a guard per defect found
- [ ] Caches redirected to `/tmp` in `%environment`
- [ ] Entry point reachable without the activation wrapper eating flags
- [ ] Docker build log captured even on failure
- [ ] SIF built from the local daemon with `-mem 512M`
- [ ] Real workload validated under `--containall --net --network=none`
- [ ] Output scored against a known reference, not just produced
- [ ] Resource ceilings measured and written into the README
- [ ] `build_env.json` and `container_validation.json` in the directory
- [ ] Push command reported to the user, not run
