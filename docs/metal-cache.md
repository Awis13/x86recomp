# Metal preparation and persistent caches

The renderer shares library compilation, pipeline descriptors and persistence
between offline preparation and live drawing. Prepared entries include checksums;
invalid entries fall back to normal compilation. Source-compiled and offline
libraries retain separate device archives. Runtime writes drain away from drawing,
first-use descriptors compile on two bounded lanes, and shutdown cancels queued
library compilation before draining archive writes.

Applications provide a `MetalPreparationProfile` from
`runtime/gfx/metal_preparation.h`. The profile supplies:

- A default cache namespace and a policy identity that changes with discovery or
  recipe rules. Policy changes invalidate preparation even when content is unchanged.
- A deterministic content identity for the selected installation. Include all
  executable, shader and configuration bytes that affect discovery.
- Shader programs, explicit pipeline recipes and any discovery errors. Recipes
  reference `MetalShaderIdentity(program)`; their serialized layout is shared with
  rendering in `metal_recipe.h`.

Discovery, file formats and named render-pass policy belong to the application.
Callbacks run synchronously before compilation and may throw for invalid inputs.
Call `ConfigureMetalPreparation(content_directory, cache_root, profile)` before
creating a device; configuration failure leaves runtime compilation available.
Use `MetalPreparationCommand(argc, argv, profile)` in the application's worker to
serve `--shader-preparation prepare|status|clean --game-dir PATH --cache-root PATH`.
The directory argument is opaque to the renderer. No application file layouts or
executable discovery rules are built into the shared worker.

Preparation requires Apple's offline Metal tools and a Metal GPU. Compilation
uses Metal 2.1 and fast math, matching runtime compilation. Local cache identity
includes source build, profile/content, OS, GPU, selected toolchain and render
settings. Compiled artifacts can be reused across source builds only when their
provenance and checksums match. Profile policy changes invalidate preparation
without discarding compatible artifacts keyed by exact source and descriptors.
Status verifies entries rather than trusting a
previous success marker. Cleaning removes only the owned `metal` subtree and
requires exclusive access.

## Tests

```sh
cmake --preset release -DRECOMP_RUNTIME_TESTS=ON -DRECOMP_METAL_TESTS=ON
cmake --build --preset release
ctest --test-dir out/build/release --output-on-failure
```

Metal integration tests need a logged-in macOS session. They use original
synthetic programs and isolated temporary caches, covering runtime persistence,
corruption, concurrent first use, source/offline origins, startup preload,
file-descriptor budgets, cancellation, profile changes and cross-build reuse.
The application-independent test profile is separate from the shipping library.
Runtime tests cover heap fragmentation/exhaustion, PE32 address-space flags,
texture-lock allocation failure and COM/WMI lifecycle behavior.
