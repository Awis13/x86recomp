#!/usr/bin/env python3
"""Observed source-IR recipes must restore off the encoder without offline tools."""
import os
from pathlib import Path
import selectors
import subprocess
import sys
import tempfile
import time
from metal_fixture import fixture

binary = str(Path(sys.argv[1]).resolve())


def release_after(process, complete):
    """Keep the real draw/readback blocked until its background work completes."""
    output = bytearray()
    selector = selectors.DefaultSelector()
    selector.register(process.stderr, selectors.EVENT_READ)
    deadline = time.monotonic() + 10
    try:
        while time.monotonic() < deadline and process.poll() is None and not complete(output):
            for key, _ in selector.select(max(0, deadline - time.monotonic())):
                chunk = os.read(key.fileobj.fileno(), 65536)
                if not chunk:
                    selector.unregister(key.fileobj)
                output.extend(chunk)
        assert complete(output), "Background pipelines did not complete:\n" + output.decode(errors="replace")
        process.stdin.write(b"\n")
        process.stdin.flush()
        stdout, stderr = process.communicate(timeout=15)
        output.extend(stderr)
        assert process.returncode == 0, stdout.decode() + output.decode()
        return output
    finally:
        selector.close()
        if process.poll() is None:
            process.kill()
            process.communicate()


with tempfile.TemporaryDirectory(prefix="recomp-recipe-reuse-") as tmp:
    base = Path(tmp)
    game = base / "game"
    effects = game
    effects.mkdir(parents=True)
    fixture(effects / "shaders.txt")
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("RECOMP_", "DYLD_"))}
    env.update(RECOMP_SHADER_CACHE_REPORT="1", DEVELOPER_DIR=str(base / "no-toolchain"))
    unused = subprocess.run([binary, str(game), str(base / "unused-cache"), "unused"],
                            env=env, capture_output=True, text=True, timeout=30)
    assert unused.returncode == 0, unused.stdout + unused.stderr
    registration = unused.stderr.split("unused begin\n")[1].split("unused end\n")[0]
    assert "shader-cache library" not in registration, (
        "Registering an unused shader loaded or compiled its library:\n" + unused.stderr)
    prefetched = subprocess.run([binary, str(game), str(base / "prefetch-cache"), "prefetch"],
                                env=env, capture_output=True, text=True, timeout=30)
    assert prefetched.returncode == 0, prefetched.stdout + prefetched.stderr
    first = subprocess.Popen([binary, str(game), str(base / "first-cache"), "first-use"], env=env,
                             stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    def first_misses(output):
        return [line for line in output.splitlines()
                if b"shader-cache pipeline miss" in line and b"phase=first-use" in line]
    output = release_after(first, lambda data: len(first_misses(data)) == 2)
    assert len(first_misses(output)) == 2, output.decode()  # no duplicate request
    assert not any(b"pipeline miss" in line and b"phase=encoder" in line
                   for line in output.splitlines()), output.decode()
    assert len(list(base.glob("first-cache/metal/v1/*/runtime-pipelines/*.metallib"))) == 2
    command = [binary, str(game), str(base / "cache")]
    seed = subprocess.run(command + ["seed"], env=env, capture_output=True, text=True, timeout=30)
    assert seed.returncode == 0, seed.stdout + seed.stderr
    assert list(base.glob("cache/metal/v1/*/runtime-pipelines/*.metallib")), seed.stderr
    assert not list(base.glob("cache/metal/v1/*/libraries/*.metallib"))
    before = {str(p): p.read_bytes() for p in base.glob("cache/metal/v1/*/runtime-pipelines/*.metallib")}
    process = subprocess.Popen(command + ["reuse"], env=env, stdin=subprocess.PIPE,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output = release_after(process, lambda data: any(b"phase=recipe" in line and
                           b"shader-cache pipeline hit" in line for line in data.splitlines()))
    assert b"shader-cache pipeline ready" in output, output.decode()
    assert b"shader-cache pipeline miss" not in output, output.decode()
    assert before == {str(p): p.read_bytes() for p in base.glob("cache/metal/v1/*/runtime-pipelines/*.metallib")}
print("Recorded first-use pipelines and observed recipe reuse verified; pixels and archives preserved")
