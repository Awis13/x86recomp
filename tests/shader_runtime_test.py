#!/usr/bin/env python3
"""Verify live Metal reuse and runtime persistence across separate processes."""
import hashlib
import fcntl
import json
import os
import re
import shutil
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from metal_fixture import fixture

RUNTIME, WORKER, NEXT_RUNTIME, NEXT_WORKER = (str(Path(p).resolve()) for p in sys.argv[1:5])
del sys.argv[1:5]


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="recomp-runtime-")
        self.base = Path(self.tmp.name)
        self.game = self.base / "game"
        effects = self.game
        effects.mkdir(parents=True)
        fixture(effects / "shaders.txt")
        self.cache = self.base / "cache"
        self.env = {k: v for k, v in os.environ.items() if not k.startswith("RECOMP_")}
        self.env["RECOMP_SHADER_CACHE_REPORT"] = "1"

    def tearDown(self):
        self.tmp.cleanup()

    def run_runtime(self, expected, executable=RUNTIME):
        result = subprocess.run([executable, str(self.game), str(self.cache), expected],
                                env=self.env, capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result.stderr

    def binaries(self):
        return {str(p.relative_to(self.cache)): (p.stat().st_mtime_ns, hashlib.sha256(p.read_bytes()).hexdigest())
                for p in self.cache.glob("metal/v1/*/*/*.metallib")}

    def shared_binaries(self):
        return {str(p.relative_to(self.cache)): (p.stat().st_mtime_ns, hashlib.sha256(p.read_bytes()).hexdigest())
                for p in self.cache.glob("metal/artifacts/v1/*/*/*.metallib")}

    def remove_artifact(self, local):
        local.unlink()
        for shared in self.cache.glob("metal/artifacts/v1/*/" + local.parent.name + "/" + local.name):
            shared.unlink()

    def prepare(self, executable=WORKER):
        result = subprocess.run([executable, "--shader-preparation", "prepare", "--game-dir", str(self.game),
                                 "--cache-root", str(self.cache)], env=self.env, capture_output=True, text=True, timeout=180)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout.splitlines()[-1])["status"], "Prepared")

    def test_profile_policy_change_uses_a_separate_cache_scope(self):
        self.prepare()
        before = self.binaries()
        scopes = set(self.cache.glob("metal/v1/*"))
        libraries = {p.name: p.stat().st_ino for p in self.cache.glob("metal/v1/*/libraries/*.metallib")}
        (self.game / "policy.txt").write_text("new recipe policy")
        status = subprocess.run([WORKER, "--shader-preparation", "status", "--game-dir", str(self.game),
                                 "--cache-root", str(self.cache)], env=self.env, capture_output=True, text=True)
        self.assertEqual(status.returncode, 0, status.stderr)
        self.assertEqual(json.loads(status.stdout)["status"], "Needs preparation")
        self.prepare()
        new_scopes = set(self.cache.glob("metal/v1/*")) - scopes
        self.assertEqual(len(new_scopes), 1)
        # Policy invalidates preparation, while identical source/descriptor keys
        # can retain their verified cross-build compiled artifacts.
        reused = {p.name: p.stat().st_ino for p in next(iter(new_scopes)).glob("libraries/*.metallib")}
        self.assertEqual(reused, libraries)
        after = self.binaries()
        self.assertTrue(all(after.get(path) == data for path, data in before.items()))

    def test_runtime_publishes_offline_libraries_without_preparation(self):
        first = self.run_runtime("runtime")
        self.assertEqual(first.count("shader-persist library saved"), 1)
        libraries = list(self.cache.glob("metal/v1/*/libraries/*.metallib"))
        self.assertEqual(len(libraries), 1)
        before = self.binaries()
        second = self.run_runtime("preloaded")
        self.assertNotIn("library miss", second)
        self.assertNotIn("pipeline miss", second)
        self.assertIn("compute hit", second)
        self.assertEqual(self.binaries(), before)
        status = subprocess.run([WORKER, "--shader-preparation", "status", "--game-dir", str(self.game),
                                 "--cache-root", str(self.cache)], env=self.env, capture_output=True, text=True)
        self.assertNotEqual(json.loads(status.stdout.splitlines()[-1])["status"], "Prepared")

    def test_runtime_archive_publication_requests_utility_priority(self):
        result = self.run_runtime("runtime")
        archives = [int(qos) for qos in re.findall(
            r"shader-persist (?:pipeline|compute) saved shutdown=0 requested_qos=(\d+)", result)]
        self.assertTrue(archives, result)
        self.assertGreaterEqual(min(archives), 0x11, "Reusable archive work remained at background QoS")

    def test_recorded_request_coalesces_callers_and_preserves_restart_reuse(self):
        self.env["DEVELOPER_DIR"] = str(self.base / "missing-xcode")
        result = self.run_runtime("requested")
        misses = [line for line in result.splitlines() if "shader-cache pipeline miss" in line]
        self.assertEqual(len(misses), 1, result)
        self.assertIn("phase=first-use", misses[0])
        self.assertEqual(len(list(self.cache.glob("metal/v1/*/runtime-pipelines/*.metallib"))), 2)  # render and compute
        before = self.binaries()
        second = self.run_runtime("archive")
        self.assertNotIn("pipeline miss", second)
        self.assertEqual(self.binaries(), before)

    def test_source_archive_hit_does_not_repeat_driver_validation_for_publication(self):
        self.env["DEVELOPER_DIR"] = str(self.base / "missing-xcode")
        self.run_runtime("runtime")
        before = self.binaries()
        second = self.run_runtime("archive")
        self.assertIn("pipeline hit", second)
        self.assertNotRegex(second, r"driver hit[^\n]*phase=persist")
        self.assertEqual(self.binaries(), before)

    def test_partial_library_publication_keeps_each_render_ir_variant(self):
        self.run_runtime("split-runtime")
        for fragment_missing in (True, False):
            with self.subTest(fragment_missing=fragment_missing):
                source = next(p for p in self.cache.rglob("sources/*.json")
                              if json.loads(p.read_text())["source"].endswith("// fragment library") == fragment_missing)
                library = source.parent.parent / "libraries" / (source.stem + ".metallib")
                self.remove_artifact(library)  # alternate both mixed source/offline origins
                before = self.binaries()
                mixed = self.run_runtime("split-archive")
                self.assertNotIn("driver rejected", mixed)
                self.assertNotIn("pipeline miss", mixed)
                after = self.binaries()
                for key, value in before.items():
                    self.assertEqual(after[key], value, "A different source/offline IR mixture replaced an archive")
                self.assertEqual(len(after), len(before) + 1)  # repaired library; mixed-IR archive already exists
                self.run_runtime("split-preloaded")
                self.assertEqual(self.binaries(), after)

    def test_incremental_entries_survive_manual_preparation(self):
        self.run_runtime("runtime")
        before = self.binaries()
        self.assertEqual(len(before), 7)  # library, four render IR origins, two compute IR origins
        self.prepare()
        prepared = self.binaries()
        for key, value in before.items():
            self.assertEqual(prepared[key], value)
        result = self.run_runtime("preloaded")
        self.assertNotIn("library miss", result)
        self.assertNotIn("pipeline miss", result)
        self.assertIn("compute hit", result)
        self.assertEqual(self.binaries(), prepared)

    def test_unavailable_toolchain_keeps_source_archive_reuse_functional(self):
        self.env["DEVELOPER_DIR"] = str(self.base / "missing-xcode")
        first = self.run_runtime("runtime")
        self.assertIn("shader-persist library failed", first)
        self.assertFalse(list(self.cache.rglob("libraries/*.metallib")))
        before = self.binaries()
        second = self.run_runtime("archive")
        self.assertIn("library miss", second)
        self.assertIn("compute hit", second)
        self.assertEqual(self.binaries(), before)
        old_scope, = (self.cache / "metal/v1").iterdir()
        del self.env["DEVELOPER_DIR"]
        self.run_runtime("runtime")  # tools select a separate scope and retry publication
        new_scope, = set((self.cache / "metal/v1").iterdir()) - {old_scope}
        self.assertTrue(list((new_scope / "libraries").glob("*.metallib")))
        self.assertFalse(list((old_scope / "libraries").glob("*.metallib")))
        self.assertEqual(len(list(self.cache.glob("metal/artifacts/v1/*/compatibility.json"))), 2)
        self.run_runtime("preloaded")
        self.env["DEVELOPER_DIR"] = str(self.base / "missing-xcode")
        self.run_runtime("archive")  # unavailable tools cannot load the offline namespace

    def test_preparation_covers_runtime_depth_copy_formats(self):
        self.prepare()
        before = self.binaries()
        result = self.run_runtime("depth-copy")
        self.assertNotIn("pipeline miss", result)
        self.assertEqual(self.binaries(), before)

    def test_recipient_toolchain_prepares_and_publishes_with_foreign_producer(self):
        # This executable carries a deliberately different kit-producer identity.
        result = self.run_runtime("runtime", NEXT_RUNTIME)
        self.assertIn("shader-persist library saved", result)
        self.assertNotIn("shader-persist library failed", result)
        self.prepare(NEXT_WORKER)
        self.run_runtime("preloaded", NEXT_RUNTIME)
        probes = [("metal", "--version"), ("--show-sdk-path",),
                  ("--show-sdk-build-version",), ("--find", "metallib")]
        identity = b"".join(subprocess.check_output(["/usr/bin/xcrun", "-sdk", "macosx", *args],
                                                  env=self.env) for args in probes)
        provenance, = self.cache.glob("metal/artifacts/v1/*/compatibility.json")
        self.assertEqual(json.loads(provenance.read_text())["toolchain"], hashlib.sha256(identity).hexdigest())

    def test_first_play_preloads_without_gameplay_recordings(self):
        self.prepare()
        self.assertFalse(list(self.cache.rglob("sources/*.json")))
        self.assertFalse(list(self.cache.rglob("descriptors/*.json")))
        before = self.binaries()
        first = self.run_runtime("firstplay")
        self.assertNotIn("pipeline miss", first)
        self.assertNotIn("library miss", first)
        self.assertEqual(self.binaries(), before)

    def test_entire_catalog_is_ready_at_startup_barrier(self):
        fixture(self.game / "shaders.txt", pixels=16)
        self.prepare()
        catalog = json.loads(next(self.cache.rglob("prepared-descriptors.json")).read_text())
        before = self.binaries()
        result = self.run_runtime("catalog")
        self.assertIn(f"shader-preload completed ready={len(catalog)} skipped=0", result)
        self.assertNotIn("pipeline miss", result)
        self.assertNotIn("library miss", result)
        self.assertEqual(self.binaries(), before)

    def test_prepared_preload_preserves_game_resource_descriptors(self):
        fixture(self.game / "shaders.txt", pixels=64)
        self.prepare()
        catalog = json.loads(next(self.cache.rglob("prepared-descriptors.json")).read_text())
        before = self.binaries()
        result = self.run_runtime("descriptor-budget")
        self.assertIn(f"shader-preload completed ready={len(catalog)} skipped=0", result)
        self.assertNotIn("library miss", result)
        self.assertNotIn("pipeline miss", result)
        self.assertEqual(self.binaries(), before)

    def test_preload_task_freezes_catalog_before_loading_screen_observations(self):
        fixture(self.game / "shaders.txt", pixels=16)
        self.prepare()
        catalog = json.loads(next(self.cache.rglob("prepared-descriptors.json")).read_text())
        before = self.binaries()
        result = self.run_runtime("catalog-snapshot")
        self.assertIn(f"shader-preload completed ready={len(catalog)} skipped=0", result)
        self.assertNotIn("pipeline miss", result)
        self.assertNotIn("library miss", result)
        self.assertEqual(self.binaries(), before)

    def test_preload_task_cancels_without_changing_prepared_binaries(self):
        self.prepare()
        before = self.binaries()
        result = self.run_runtime("cancel-preload")
        self.assertIn("cancelled", result)
        self.assertNotIn("pipeline miss", result)
        self.assertNotIn("library miss", result)
        self.assertEqual(self.binaries(), before)

    def test_corrupt_catalog_archive_is_skipped_without_compilation(self):
        self.prepare()
        catalog_file = next(self.cache.rglob("prepared-descriptors.json"))
        catalog = json.loads(catalog_file.read_text())
        key = next(k for k, record in catalog.items() if not record.get("core"))
        archive = catalog_file.parent / "pipelines" / (key + ".metallib")
        # A bad checksum on an otherwise Metal-valid archive must also reject
        # preload: driver rejection alone would not exercise that requirement.
        for target in [archive, Path(str(archive) + ".sha256")]:
            with self.subTest(entry=target.suffix):
                saved = target.read_bytes()
                target.write_bytes(b"corrupt entry")
                try:
                    before = self.binaries()
                    result = self.run_runtime("firstplay")
                    self.assertIn(f"shader-preload completed ready={len(catalog) - 1} skipped=1", result)
                    self.assertNotIn("driver miss", result)
                    self.assertNotIn("library miss", result)
                    self.assertEqual(self.binaries(), before)
                    self.assertEqual(target.read_bytes(), b"corrupt entry")
                finally:
                    target.write_bytes(saved)

    def test_first_play_observations_can_be_prepared_without_source_records(self):
        self.prepare()
        first = self.run_runtime("firstplay-record")
        self.assertFalse(list(self.cache.rglob("sources/*.json")))
        self.assertTrue(list(self.cache.rglob("descriptors/*.json")))
        self.prepare()
        second = self.run_runtime("firstplay")
        ready = lambda log: int(re.search(r"shader-preload completed ready=(\d+)", log)[1])
        self.assertLess(ready(second), ready(first), "Later startup still loads the entire unused baseline")

    def test_corrupt_first_play_catalog_skips_preload_without_compilation(self):
        self.prepare()
        catalog = next(self.cache.rglob("prepared-descriptors.json"))
        catalog.write_text("corrupt metadata")
        before = self.binaries()
        result = self.run_runtime("firstplay-archive")
        self.assertIn("shader-preload completed ready=0 skipped=0", result)
        self.assertNotIn("pipeline miss", result)
        self.assertEqual(self.binaries(), before)

    def test_archive_only_does_not_compile_and_corruption_falls_back(self):
        missing = self.run_runtime("unavailable")
        self.assertIn("pipeline unavailable", missing)
        self.assertFalse(list(self.cache.rglob("*pipelines/*.metallib")))
        self.run_runtime("runtime")
        self.prepare()
        archives = list(self.cache.rglob("pipelines/*.metallib"))
        for archive in archives:
            archive.write_bytes(b"corrupted entry")
        fallback = self.run_runtime("runtime")
        self.assertRegex(fallback, r"shader-preload completed ready=0 skipped=[1-9]\d*")
        self.assertIn("shader-persist pipeline saved", fallback)
        self.run_runtime("preloaded")

    def test_missing_library_skips_preload_without_compilation(self):
        self.run_runtime("runtime")
        self.prepare()
        source = next(self.cache.rglob("sources/*.json"))
        library = source.parent.parent / "libraries" / (source.stem + ".metallib")
        self.remove_artifact(library)
        result = self.run_runtime("archive")
        self.assertRegex(result, r"shader-preload completed ready=\d+ skipped=1")
        self.assertIn("library miss", result)
        self.assertTrue(library.exists())  # automatic publication repairs the missing library

    def test_archive_write_failures_and_worker_lock_do_not_break_drawing(self):
        self.cache.mkdir()
        lock = self.cache / "metal.lock"
        with lock.open("w") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            result = self.run_runtime("runtime")
        self.assertIn("shader-persist library busy", result)
        self.assertIn("shader-persist pipeline busy", result)
        self.assertEqual(self.binaries(), {})
        self.run_runtime("unavailable")  # publish just the offline library
        scope = next((self.cache / "metal/v1").iterdir())
        (scope / "pipelines").write_text("not a directory")
        failed = self.run_runtime("runtime")
        self.assertIn("shader-persist pipeline failed", failed)
        (scope / "pipelines").unlink()
        self.run_runtime("runtime")
        self.run_runtime("preloaded")

    def test_library_write_failure_keeps_render_and_compute_functional(self):
        self.run_runtime("runtime")
        scope = next((self.cache / "metal/v1").iterdir())
        for library in (scope / "libraries").glob("*.metallib"):
            self.remove_artifact(library)
        shutil.rmtree(scope / "libraries")
        (scope / "libraries").write_text("not a directory")
        result = self.run_runtime("archive")
        self.assertIn("shader-persist library failed", result)
        self.assertNotIn("compute miss", result)
        (scope / "libraries").unlink()
        self.run_runtime("archive")
        # Source archive hits also publish the offline-IR versions after repair.
        self.run_runtime("preloaded")

    def test_library_backlog_allows_pipeline_writes_between_compiles(self):
        result = self.run_runtime("queue-order")
        self.assertEqual(result.count("shader-persist library saved"), 9)
        self.assertLess(result.index("shader-persist pipeline saved"),
                        result.rindex("shader-persist library saved"))

    def test_source_metadata_precedes_descriptors_despite_library_backlog(self):
        self.run_runtime("metadata-order")
        descriptor, = self.cache.glob("metal/v1/*/descriptors/*.json")
        record = json.loads(descriptor.read_text())
        scope = descriptor.parent.parent
        for stage in ("vertex", "fragment"):
            source = scope / "sources" / (record[stage].split("/")[0] + ".json")
            self.assertTrue(source.exists(), stage + " source metadata was lost")
            self.assertLess(source.stat().st_mtime_ns, descriptor.stat().st_mtime_ns,
                            stage + " source was published after its descriptor; interruption leaves an orphan")

    def test_shutdown_defers_pending_libraries_without_leaking_temporary_files(self):
        result = self.run_runtime("cancel-pending")
        self.assertIn("shader-persist library deferred", result)
        continuations = [int(qos) for qos in re.findall(
            r"shader-persist library deferred shutdown=1 requested_qos=(\d+)", result)]
        self.assertTrue(continuations, result)
        self.assertGreaterEqual(min(continuations), 0x19, "Shutdown continuations returned to background QoS")
        self.assertEqual(len(list(self.cache.rglob("sources/*.json"))), 65)
        self.assertFalse(list(self.cache.rglob("*.metal")))
        self.assertFalse(list(self.cache.rglob("*.air")))
        self.assertFalse(list(self.cache.rglob("state.json")))
        # Encountered sources survive cancellation; a new process retries what
        # it uses, without launching discovery of the rest of the installation.
        self.run_runtime("resume")
        self.run_runtime("preloaded")

    def test_background_archive_queue_owns_entry_keys(self):
        result = self.run_runtime("stress")
        self.assertNotIn("shader-persist pipeline failed", result)
        self.assertGreaterEqual(result.count("shader-persist pipeline saved"), 100)

    def test_normal_process_exit_drains_archive_writes(self):
        result = self.run_runtime("stress-exit")
        self.assertGreaterEqual(result.count("shader-persist pipeline saved"), 100)
        self.assertGreaterEqual(len(self.binaries()), 100)
        # Library continuations can enqueue work behind the first queue drain.
        # Normal exit must retain all encountered sources and finish archives.
        self.assertEqual(len(list(self.cache.rglob("sources/*.json"))), 9)
        self.assertFalse(list(self.cache.rglob("*.air")))
        self.assertFalse(list(self.cache.rglob("*.metal")))
        before = self.binaries()
        resumed = self.run_runtime("resume")
        self.assertNotIn("pipeline miss", resumed)
        after = self.binaries()
        for key, value in before.items():
            self.assertEqual(after[key], value)

    def test_new_build_reuses_binaries_without_inheriting_preparation(self):
        self.run_runtime("runtime")
        self.prepare()
        old_scope = next((self.cache / "metal/v1").iterdir())
        shared = self.shared_binaries()
        self.assertTrue(shared, "Shared artifact store was not enabled")
        log = self.run_runtime("archive", NEXT_RUNTIME)
        self.assertNotIn("library miss", log)
        self.assertNotIn("pipeline miss", log)
        self.assertIn("compute hit", log)
        new_scope, = set((self.cache / "metal/v1").iterdir()) - {old_scope}
        self.assertFalse((new_scope / "state.json").exists())
        self.assertFalse((new_scope / "translations").exists())
        self.assertFalse((new_scope / "metal-msl-v1.bin").exists())
        self.assertFalse((new_scope / "prepared-descriptors.json").exists())
        self.assertFalse((new_scope / "libraries").exists(), "Drawing materialized shared binaries")
        self.assertTrue(list((new_scope / "sources").glob("*.json")))
        status = subprocess.run([NEXT_WORKER, "--shader-preparation", "status", "--game-dir", str(self.game),
                                 "--cache-root", str(self.cache)], env=self.env, capture_output=True, text=True, check=True)
        self.assertEqual(json.loads(status.stdout.splitlines()[-1])["status"], "Needs preparation")
        self.prepare(NEXT_WORKER)
        self.assertEqual(self.shared_binaries(), shared, "Unchanged artifacts were rebuilt or rewritten")
        manifest = json.loads((new_scope / "state.json").read_text())
        self.assertEqual(manifest["status"], "Prepared")
        self.assertTrue((new_scope / "translations").is_dir())
        for relative, checksum in manifest["entries"].items():
            self.assertFalse(Path(relative).is_absolute())
            self.assertNotIn("..", Path(relative).parts)
            entry = new_scope / relative
            self.assertEqual(hashlib.sha256(entry.read_bytes()).hexdigest(), checksum)
            if entry.suffix == ".metallib":
                shared_entry, = self.cache.glob("metal/artifacts/v1/*/" + entry.parent.name + "/" + entry.name)
                self.assertTrue(entry.samefile(shared_entry), "Preparation duplicated a shared binary")
        self.run_runtime("preloaded", NEXT_RUNTIME)

    def test_changed_settings_keep_shared_stores_isolated(self):
        self.run_runtime("runtime")
        old_shared = self.shared_binaries()
        self.env["RECOMP_RENDER_SCALE"] = "0.5"
        log = self.run_runtime("runtime", NEXT_RUNTIME)
        self.assertIn("library miss", log)
        self.assertIn("pipeline miss", log)
        self.assertEqual(len(list(self.cache.glob("metal/artifacts/v1/*/compatibility.json"))), 2)
        for name, value in old_shared.items():
            self.assertEqual(self.shared_binaries()[name], value)

    def test_new_build_reuses_shared_source_ir_without_offline_toolchain(self):
        self.env["DEVELOPER_DIR"] = str(self.base / "missing-xcode")
        self.run_runtime("runtime")
        shared = self.shared_binaries()
        self.assertTrue(shared)
        log = self.run_runtime("archive", NEXT_RUNTIME)
        self.assertIn("library miss", log)
        self.assertNotIn("pipeline miss", log)
        self.assertIn("compute hit", log)
        self.assertEqual(self.shared_binaries(), shared)
        self.assertFalse(list(self.cache.rglob("libraries/*.metallib")))

    def test_valid_shared_archive_with_wrong_functions_requires_driver_fallback(self):
        self.run_runtime("runtime")
        descriptor = next(self.cache.glob("metal/v1/*/descriptors/*.json"))
        render, = self.cache.glob("metal/artifacts/v1/*/pipelines/" + descriptor.stem + ".metallib")
        compute, = (p for p in render.parent.glob("*.metallib") if p != render)
        # The archive loads successfully and its checksum is correct, but it
        # contains compute functions rather than the requested render pipeline.
        render.write_bytes(compute.read_bytes())
        Path(str(render) + ".sha256").write_text(hashlib.sha256(render.read_bytes()).hexdigest())
        log = self.run_runtime("runtime", NEXT_RUNTIME)
        self.assertNotIn("library miss", log)
        self.assertIn("driver rejected", log)
        self.assertIn("pipeline miss", log)
        self.assertIn("compute hit", log)
        self.run_runtime("preloaded", NEXT_RUNTIME)

    def test_shared_driver_rejection_falls_back_and_repairs(self):
        self.run_runtime("runtime")
        old_scope = next((self.cache / "metal/v1").iterdir())
        # Valid checksums alone must never authorize reuse of an unloadable file.
        for entry in self.cache.glob("metal/artifacts/v1/*/*/*.metallib"):
            entry.write_bytes(b"not a Metal binary")
            Path(str(entry) + ".sha256").write_text(hashlib.sha256(entry.read_bytes()).hexdigest())
        log = self.run_runtime("runtime", NEXT_RUNTIME)
        self.assertIn("library miss", log)
        self.assertIn("pipeline miss", log)
        self.assertIn("compute miss", log)
        self.run_runtime("preloaded", NEXT_RUNTIME)
        self.assertTrue(old_scope.exists())

    def test_opaque_legacy_scopes_are_not_pooled(self):
        self.run_runtime("runtime")
        old_scope = next((self.cache / "metal/v1").iterdir())
        old_scope.rename(old_scope.with_name("opaque-legacy-scope"))
        shutil.rmtree(self.cache / "metal/artifacts")
        log = self.run_runtime("runtime", NEXT_RUNTIME)
        self.assertIn("library miss", log)
        self.assertIn("pipeline miss", log)
        self.assertTrue((old_scope.parent / "opaque-legacy-scope").exists())

    def test_shared_publication_failure_keeps_local_cache_functional(self):
        owned = self.cache / "metal"
        owned.mkdir(parents=True)
        (owned / "artifacts").write_text("blocked shared store")
        log = self.run_runtime("runtime")
        self.assertIn("sharing compiled shader artifact", log)
        self.assertNotIn("shader-persist library failed", log)
        self.run_runtime("preloaded")


if __name__ == "__main__":
    unittest.main()
