# Swift Export Rollout — Recipe for the Claude GitHub Agent

> **Self-contained instruction set.** This file lives at the root of every
> repository that has the canonical KotlinMania Swift Export rollout
> committed to its `automation/swift-export-rollout` branch. The Claude
> GitHub Agent will be assigned a per-repo issue that asks it to walk this
> recipe top-to-bottom and finish the rollout by pushing the branch and
> opening the pull request. Nothing in this recipe depends on context from
> outside the current repository.

## TL;DR

A workspace-wide rollout already produced a local branch named
`automation/swift-export-rollout` with a single canonical commit (the
"rollout commit") that wires the Kotlin → Swift Export → SPM → swift-test
loop into the repository. **Your job, Claude, is to:**

1. **Push** the local `automation/swift-export-rollout` branch to `origin`.
2. **Open a pull request** against the base branch the rollout was made
   from (NOT necessarily `main`), using the title and body template in
   [§ PR template](#pr-template) below.
3. **Verify** the PR landed cleanly and CI was kicked off.
4. **Comment on the assigned issue** linking the new PR and close the
   issue with a one-line summary.

Do **NOT** modify any source files. Do **NOT** rebase, squash, or amend
the rollout commit. Do **NOT** change the branch base. If anything in
the precondition checks fails, [stop and report](#what-to-do-when-the-recipe-can-not-be-followed)
rather than improvising.

---

## Background — what's already on this branch

The rollout commit applies a strict, idempotent set of five changes
that are byte-identical across every KotlinMania repository it touched.
These were produced by `automation-artifacts/swift-export-rollout/apply.sh`
in the workspace, mirroring the canonical pattern that landed in
[`libc-kotlin`](https://github.com/KotlinMania/libc-kotlin) and was
proven end-to-end with a working `swift test` invocation against the
`embedSwiftExportForXcode`-produced Swift Package Manager package.

### The five changes

1. **`build.gradle.kts`** — **both** iOS Simulator framework binaries
   (`iosSimulatorArm64` *and* `iosX64`) now have `isStatic = true`.
   The Swift Export SPM bridge expects a static framework for the iOS
   Simulator so that the SPM linker can pull in the symbol
   implementations directly. Both simulator slices must match — if
   `iosSimulatorArm64` is static and `iosX64` is left dynamic, the
   `assembleDebugIosSimulatorFatFrameworkFor<Name>XCFramework` task
   fails with:

   ```
   Cannot create a fat framework from:
     <Name> - arm64 - static
     <Name> - x64 - dynamic
     All input frameworks must be either static or dynamic
   ```

   The other Apple targets (`iosArm64`, `tvosArm64`,
   `tvosSimulatorArm64`, `watchos*`, `macosArm64`) stay dynamic.

   > **Note**: earlier versions of this recipe and of `apply.sh` flipped
   > only `iosSimulatorArm64`. Repos that landed the rollout before this
   > correction need a follow-up commit adding `isStatic = true` to
   > `iosX64`. http-kotlin's `build.gradle.kts:218-231` is the working
   > reference shape.

2. **`.github/workflows/swift.yml`** — a new platform workflow with a
   `workflow_call:` trigger, matching the shape of the existing
   `macos.yml` / `ios.yml` / `tvos.yml` / etc. workflows. It:
     - Sets every Xcode-style environment variable that the Kotlin
       Multiplatform plugin's `BuildSPMSwiftExportPackage` task reads
       as input. The list includes `BUILT_PRODUCTS_DIR`,
       `TARGET_BUILD_DIR`, `SDK_NAME=macosx`, `CONFIGURATION=Debug`,
       `ARCHS=arm64`, `FRAMEWORKS_FOLDER_PATH=Frameworks`,
       `MACOSX_DEPLOYMENT_TARGET=14.0`, and crucially
       `DEPLOYMENT_TARGET_SETTING_NAME=MACOSX_DEPLOYMENT_TARGET` —
       which Xcode normally injects per-target but the Kotlin plugin
       has no default for. Without it,
       `BuildSPMSwiftExportPackage` fails its property validation
       with `property 'deploymentTargetSettingName' doesn't have a
       configured value`.
     - Runs `./gradlew embedSwiftExportForXcode --no-configuration-cache`
       to populate `build/swift-test/` with a `lib<Module>.a` static
       archive containing every `@ExportedBridge`-annotated function
       generated from the upstream Kotlin source, plus four
       `.swiftmodule/` bundles named `<Module>`, `KotlinRuntimeSupport`,
       `ExportedKotlinPackages`, and `KotlinRuntime`.
     - Changes directory into `swift-test-harness/` and runs
       `swift test`, which compiles and runs the smoke test against
       the just-produced Swift module.
     - Uploads the static archive, swiftmodule bundles, and the full
       generated SPM package tree as a `swift-export-artifacts` build
       artifact on both success and failure so a broken run is
       debuggable from the GitHub Actions run summary.

3. **`.github/workflows/ci.yml`** — a four-line `swift:` job is wired
   in next to the existing `wasm:` and `js:` orchestrator entries with
   the same `permissions: contents: read` ceiling. This makes the
   `swift.yml` workflow run as part of the regular CI orchestration.

4. **`swift-test-harness/`** — a new top-level directory containing:
     - `Package.swift` — a Swift Package Manager package depending on
       the Kotlin-generated SPM at
       `../build/SPMPackage/macosArm64/Debug` with one `testTarget`
       (`SwiftTestHarnessTests`). The testTarget carries a
       `linkerSettings` block with
       `unsafeFlags(["-L", "../build/swift-test", "-l<Module>"])` so
       that `swift test` (run outside an `xcodebuild`-driven
       invocation) can resolve the `__root____*` and KotlinError
       symbols from the static archive that `embedSwiftExportForXcode`
       drops next to the Sources/ tree. This linker hint is necessary
       because the Kotlin plugin's generated `Package.swift` does NOT
       reference its own `lib<Module>.a` as a binary target (one of
       the two upstream gaps noted in [§ Known upstream gaps](#known-upstream-gaps)
       below).
     - `Tests/SwiftTestHarnessTests/<Module>ExportTests.swift` — a
       smoke test that does `import <Module>` and asserts a single
       `XCTAssertTrue(true, ...)`. Because the import is at the
       module level, the file's mere ability to compile proves that
       `<Module>.swiftmodule` exists and is correctly placed; the
       fact that the test executable links proves the static archive
       is reachable; the fact that the test runs proves the full
       loop is green for this repo.
     - `.gitignore` — ignores the SPM `.build/` output directory.

5. **`.gitignore`** — appended with a Swift/Xcode workspace state
   section: `.swiftpm/`, `DerivedData/`, `*.xcuserstate`,
   `*.xcworkspace/xcuserdata/`, `*.xcodeproj/xcuserdata/`. The
   Kotlin Swift Export staging directory `build/swift-test/` is
   already covered by the existing `build/` rule.

### Background — what Swift Export does

Kotlin 2.2+ ships an experimental Swift Export feature
(`kotlin.experimental.swift-export.enabled`) that emits idiomatic
Swift bindings for a Kotlin Multiplatform module's exported API.
Instead of consuming the Kotlin output through Objective-C interop
(the legacy approach that produces `KotlinInt`, mangled names, and
`underscore_prefixed_calls`), Swift consumers get a real
`.swiftmodule` they can `import <Module>` against, with Kotlin
typealiases mapped to native Swift types, generic parameters
preserved, and nullability flowing through correctly.

The KotlinMania workspace standardized on Swift Export across every
repository that ships an XCFramework so that downstream Swift / Xcode
consumers don't have to opt into the legacy interop. The rollout
this recipe describes is the mechanical part — wiring up the build,
the test harness, and the CI workflow.

---

## Preconditions to verify before acting

Run these checks in order. If any of them fails, stop and report —
**do NOT attempt to fix the repository state**; the workspace-level
rollout will be re-run instead.

### 1. The rollout branch exists locally

```sh
git rev-parse --verify automation/swift-export-rollout
```

Must print a commit SHA. If this errors with "unknown revision", the
rollout never landed in this repo and there's nothing to push.

### 2. The rollout commit is the tip of the branch

```sh
git log -1 --format='%s' automation/swift-export-rollout
```

Must print exactly:

```
Wire Kotlin → Swift Export → swift test loop into CI
```

If the tip commit's subject is something else, someone has amended,
rebased, or added commits on top — stop and report.

### 3. The rollout files are present at the tip

```sh
git show --stat automation/swift-export-rollout
```

Must list at minimum:

- `build.gradle.kts` modified (1 insertion if multi-line iosSimulatorArm64 form, 4 insertions and 1 deletion if single-line form)
- `.github/workflows/swift.yml` new
- `.github/workflows/ci.yml` modified (4-line addition)
- `swift-test-harness/Package.swift` new
- `swift-test-harness/.gitignore` new
- `swift-test-harness/Tests/SwiftTestHarnessTests/<Module>ExportTests.swift` new
- `.gitignore` modified

It may also include this `SWIFT_EXPORT_ROLLOUT.md` file at the repo
root — that's the recipe being committed alongside the rollout.

If any of the seven (or eight, counting this recipe) files are missing
from the diff, stop and report.

### 4. The base branch is known

The rollout branch was created off the repo's HEAD at the time of the
rollout — that's the **base branch**, and that's where the PR should
target. Determine it by walking back through the local reflog:

```sh
git reflog show automation/swift-export-rollout | tail -5
```

The earliest entry's "from" branch is the base. In most repos this
will be `main`. In a small handful it will be a feature branch like
`setup-ci`, `automation/kotlinmania-ci-h14-4`, or
`port/drop-bignum-add-build-gate` (these are repos that had
in-progress work at rollout time; the rollout was branched off the
feature branch to keep scopes clean).

If you cannot determine the base from the reflog, default to `main`
**but** flag this in your final summary so the human reviewer can
verify.

### 5. Origin remote exists and points at GitHub

```sh
git remote get-url origin
```

Must be a `github.com/KotlinMania/<repo>` URL (HTTPS or SSH).

---

## The work

### Step 1 — push the rollout branch

```sh
git push -u origin automation/swift-export-rollout
```

Expected outcome: the remote ref is created (or fast-forwarded if
it already existed from a previous attempt). If the push is rejected
because the remote ref exists and has diverged, **do not force-push**;
stop and report.

### Step 2 — open the pull request

Use `gh pr create` with the title and body from
[§ PR template](#pr-template) below. The base branch is the one you
determined in precondition #4.

```sh
gh pr create \
    --base "<base-branch>" \
    --head "automation/swift-export-rollout" \
    --title "Wire Kotlin → Swift Export → swift test loop into CI" \
    --body-file <(cat <<'BODY'
... see § PR template ...
BODY
)
```

Capture the returned PR URL — you'll need it for the issue comment.

### Step 3 — verify CI kicked off

```sh
gh pr view <pr-number> --json statusCheckRollup
```

If the field is empty or shows "queued" / "in_progress", that's
expected — the workflow_call orchestration may take a minute to fan
out. Don't wait for completion; just verify a run was registered.

If the PR was created against a base branch that does NOT have CI
configured to run on PRs (e.g., a `chore/*` feature branch with no
PR triggers), there will be no check rollup. That's still fine —
note it in the summary.

### Step 4 — close the assigned issue

Post a comment on the assigned issue (the one that asked you to do
this work) with the format below, then close the issue.

```
Pushed `automation/swift-export-rollout` and opened PR:
<pr-url>

PR base: <base-branch>
CI status: <queued | in_progress | n/a — no PR triggers on base>
Rollout commit: <commit-sha>

Done.
```

---

## PR template

### Title

```
Wire Kotlin → Swift Export → swift test loop into CI
```

### Body

```markdown
## Summary

This PR wires the Kotlin → Swift Export → SPM → `swift test` loop into
this repository's CI, mirroring the canonical pattern established by
[libc-kotlin](https://github.com/KotlinMania/libc-kotlin) and proven
end-to-end with a passing `swift test` invocation against the
`embedSwiftExportForXcode`-produced Swift Package Manager package.

The full recipe for the rollout — including the rationale for each
change, the two upstream Kotlin Multiplatform plugin gaps it works
around, and the precondition / verification checks the Claude agent
ran before opening this PR — is checked in at the repo root as
[`SWIFT_EXPORT_ROLLOUT.md`](./SWIFT_EXPORT_ROLLOUT.md).

## What this PR changes

- `build.gradle.kts` — `iosSimulatorArm64` framework binary now has
  `isStatic = true` so the Swift Export SPM bridge can link against
  it.
- `.github/workflows/swift.yml` (new) — `workflow_call:` platform
  workflow that sets the full Xcode-style environment, runs
  `./gradlew embedSwiftExportForXcode`, then runs `swift test` from
  `swift-test-harness/`. Uploads the static archive and `.swiftmodule`
  bundles as artifacts on success-or-failure.
- `.github/workflows/ci.yml` — adds the `swift:` job alongside
  `wasm:` and `js:`.
- `swift-test-harness/` (new) — a Swift Package Manager package with
  one smoke test that asserts the Swift module imports cleanly.
- `.gitignore` — Swift/SPM workspace state section.

## Test plan

- [ ] The `Build (Swift)` job in CI lands BUILD SUCCESSFUL.
- [ ] The `Run swift test against Kotlin-exported module` step prints
      1 test executed, 0 failures.
- [ ] The uploaded `swift-export-artifacts` archive contains
      `lib<Module>.a` and the four `.swiftmodule/` directories.

## Known upstream gaps

Two issues live in the upstream Kotlin Multiplatform plugin and are
worked around per-repo in this PR. Both would, if fixed upstream,
allow every Kotlin Swift Export consumer to drop the local workaround
at once:

1. **`Package.swift` doesn't reference its own static archive.** The
   Kotlin plugin emits Swift source files for the SPM package but
   does not declare `lib<Module>.a` as a binary target or expose its
   link path. `swift-test-harness/Package.swift` compensates with
   `unsafeFlags(["-L", "../build/swift-test", "-l<Module>"])`.
2. **`BuildSPMSwiftExportPackage` validates
   `deploymentTargetSettingName` as a non-optional input** but
   provides no default. Outside Xcode the env var
   `DEPLOYMENT_TARGET_SETTING_NAME` isn't set, so the task fails
   property validation. `swift.yml` sets it explicitly.

🤖 Opened by the Claude GitHub Agent following
[`SWIFT_EXPORT_ROLLOUT.md`](./SWIFT_EXPORT_ROLLOUT.md).
```

---

## What to do when the recipe can NOT be followed

If any precondition check fails, or if any step errors out in a way
not anticipated by the recipe, **do NOT improvise**. Specifically:

- Do NOT force-push.
- Do NOT amend, rebase, or squash the rollout commit.
- Do NOT modify any source files.
- Do NOT change the PR base.
- Do NOT close the issue without a PR link, unless the rollout truly
  doesn't apply to this repo (e.g., no XCFramework declaration, no
  Apple targets).

Instead, post a comment on the assigned issue with the format:

```
Cannot complete rollout — <one-line summary>.

What I observed:
<copy/paste the relevant `git`, `gh`, or shell output>

What this implies:
<short interpretation>

Leaving the issue open for a human to follow up. The local
`automation/swift-export-rollout` branch (if it exists) has not
been modified.
```

Then stop. A human will pick it up.

---

## Known upstream gaps

1. **Kotlin-plugin-generated `Package.swift` doesn't reference its
   static archive.** Workaround in `swift-test-harness/Package.swift`:
   `unsafeFlags(["-L", "../build/swift-test", "-l<Module>"])`.

2. **`BuildSPMSwiftExportPackage.deploymentTargetSettingName` has no
   default.** Workaround in `.github/workflows/swift.yml`:
   `DEPLOYMENT_TARGET_SETTING_NAME=MACOSX_DEPLOYMENT_TARGET` in the
   env block for the macOS build.

3. **`KotlinStdlib.kt` unchecked-cast bridge** (the big one). When a
   declaration that reaches the Swift boundary exposes
   `kotlin.Result<T>` or `kotlin.Throwable` / `kotlin.Exception`, the
   Kotlin plugin generates
   `build/SwiftExport/<target>/<config>/files/KotlinStdlib/KotlinStdlib.kt`
   containing `Any?` → `Array<Any?>` unchecked casts. Under the
   workspace-canonical `allWarningsAsErrors.set(true)` these become
   compile errors.

   **First-choice fix when Swift does not need that surface:** keep the
   Kotlin declaration public, but hide only the Swift-hostile member or
   type with `@HiddenFromObjC` and add the required file opt-in:

   ```kotlin
   @file:OptIn(kotlin.experimental.ExperimentalObjCRefinement::class)

   package io.github.kotlinmania.example

   import kotlin.native.HiddenFromObjC

   @HiddenFromObjC
   public class ParseError(/* ... */) : Exception()

   public class Parser {
       @HiddenFromObjC
       public fun tryParse(input: String): Result<Value> =
           parseValue(input).fold(
               onSuccess = { Result.success(it) },
               onFailure = { Result.failure(ParseError(it.message)) },
           )
   }
   ```

   `@HiddenFromObjC` affects only the Kotlin/Native Obj-C / Swift bridge;
   Kotlin callers, including sibling `*-kotlin` repos, continue to see the
   same public API. This is the correct repair when the Kotlin port must
   preserve upstream-shaped public signatures but the Swift smoke test only
   needs the rest of the module to import and link. uuid-kotlin PR #13 used
   this pattern for public `Result<Uuid>` parser helpers plus public
   `Exception`-derived parse error types.

   **When Swift consumers legitimately need the failing result surface:**
   replace `kotlin.Result<T>` in that public API with a repo-local concrete
   result type using the flat-class pattern below. See
   [`triage-kotlin-stdlib-in-public-api.md`](./triage-kotlin-stdlib-in-public-api.md)
   for the workspace-wide hit list.

4. **Swift Export does NOT preserve Kotlin sealed-subclass casts.**
   Discovered empirically while making http-kotlin's `swift test`
   actually link. A `sealed class Foo { class Ok(...) : Foo(); class
   Err(...) : Foo() }` does not expose `Http.FooOk` / `Http.FooErr`
   as Swift types that consumers can reach with `as? Http.FooOk`.
   The Swift consumer can `import` `Foo` but cannot pattern-match on
   its sealed subclasses. **Use a flat class with nullable fields
   and boolean predicates instead** — see "Recipe for replacing
   `kotlin.Result<T>` in a public API" below.

5. **Swift accesses Kotlin companion objects as `Type.Companion.shared.member`**,
   not `Type.companion.member`. Common gotcha when writing the smoke
   test in `swift-test-harness/`. The Swift binding generated by the
   plugin uses a capital `Companion` struct type with a `shared`
   singleton accessor, not a lowercase `companion` property.

6. **Mutable generic collections in public APIs crash the bridging code.**
   `MutableList<T>` / `MutableSet<T>` / `MutableMap<K, V>` as public
   property types or return types cause the Swift Export compiler
   plugin to crash while emitting Swift-to-Kotlin bridging code,
   typically with type mismatch errors between `MutableList<Any?>`
   and `MutableList<T>`. The plugin's generic erasure to upper bound
   (`Any?`) collides with the mutability covariance constraints.
   **Per-repo workaround**: change public properties and return types
   to their immutable equivalents (`List<T>`, `Set<T>`, `Map<K, V>`).
   Where mutability is genuinely needed internally, hide the mutable
   collection with `internal` and expose the read-only view publicly.

7. **`macos-latest` ships an SDK older than the Kotlin/Native platform
   caches reference.** The Kotlin/Native 2.3.x distribution embeds
   prebuilt caches —
   `liborg.jetbrains.kotlin.native.platform.{CoreFoundation,Foundation,darwin}-cache.a`
   — that reference macOS 26 / iOS 26 SDK symbols
   (`_NSCalendarIdentifier{Vikram,Bangla,Dangi,Gujarati,…}`,
   `_NSURLUbiquitousItem{IsSyncPaused,SupportedSyncControls}Key`,
   `_mach_vm_reclaim_update_kernel_accounting_trap`,
   `_xpc_{connection,session,listener}_set_peer_requirement`,
   `_kCF{Bangla,Dangi,…}Calendar`). On GitHub Actions, `macos-latest`
   currently resolves to `macos-15` with Xcode 16 / macOS 15 SDK, so
   `swift test` link fails with "Undefined symbols for architecture
   arm64" on every symbol the cache references but the SDK doesn't
   carry. A consumer whose Kotlin code never reaches into those APIs
   may not notice (libc-kotlin happens to be in this lucky bucket),
   but any consumer that does (schemars-kotlin et al.) link-fails.
   **Workaround**: in `.github/workflows/swift.yml` set
   `runs-on: macos-26`. The macos-26 image's default Xcode is in the
   26.x line (carries 26.0.1 / 26.1.1 / 26.2 / 26.3 / 26.4.1 / 26.5;
   26.4.1 is the image default at the time of writing) and supplies
   the SDK stubs the platform cache expects.

   > **Do NOT add a `maxim-lobanov/setup-xcode@v1` step.** Earlier
   > revisions of this recipe pinned the Xcode version explicitly via
   > that third-party action. KotlinMania's enterprise GitHub Actions
   > policy does not whitelist third-party actions, and any workflow
   > that references one returns `startup_failure` at dispatch time
   > with zero jobs spawned — confirmed across two consecutive
   > dispatches on itertools-kotlin/automation/swift-export-rollout
   > (runs `26326086457`, `26326121639`). The pin step was redundant
   > anyway: `swift --version` from a working run on macos-26 prints
   > `Target: arm64-apple-macosx26.0` and `Xcode 26.4.1` straight from
   > the image default. Keep `runs-on: macos-26`; omit the pin step.

These first seven gaps are filable upstream against the Kotlin Multiplatform plugin
(or, for #7, against the Kotlin/Native distribution's platform-cache
build pipeline). When they're fixed, the workarounds in this recipe
become deletable.

8. **The Kotlin Swift Export bridge file casts `Any` / `Any?` to typed
   Kotlin values under `allWarningsAsErrors=true`.** The plugin-generated
   bridge at `build/SwiftExport/<target>/<config>/files/<Module>/<Module>.kt`
   pushes every typed Kotlin value through an untyped `Any` channel
   and then casts back to the concrete type on the way out. Under the
   workspace-canonical `allWarningsAsErrors.set(true)` every such cast
   trips `Unchecked cast` and `compileSwiftExportMainKotlin<Target>`
   fails. There are **two distinct API shapes** that provoke this; the
   bridge file looks similar in both cases but the source-side fix is
   different.

   **Trigger 8a — unconstrained generic public classes.** When a
   public Kotlin API exposes a generic class whose type parameter is
   unconstrained (`WithPosition<T>`, `ZipEq<L, R>`, `Unfold<St, T>`,
   etc.), the plugin erases `T` to `kotlin.Any?` in the bridge and
   emits:

   ```kotlin
   val ____self = dereferenceExternalRCRef(self) as io.github.kotlinmania.<pkg>.ZipEq<kotlin.Any?, kotlin.Any?>
   // w: Unchecked cast: 'Any?' to 'ZipEq<Any?, Any?>'.
   ```

   Same shape as gap #3 but for generic public types rather than
   `kotlin.Result`. The flat-class workaround in [§ Recipe for replacing
   `kotlin.Result<T>` in a public API](#recipe-for-replacing-kotlinresultt-in-a-public-api)
   does NOT apply — these are iterator/wrapper types that cannot be
   flattened without unfaithful translation.

   *Per-repo fix:* stop exposing the unconstrained generic classes.
   Swift Export only emits bridge code for symbols visible in the
   public API surface — internal types are skipped, so no
   `as <UserClass><Any?>` cast gets generated for them. The pattern
   from itertools-kotlin commit `40513a5`:

   - Mark every public unconstrained generic iterator/wrapper class
     `internal class` (the class itself, its companion factories, and
     any nested sealed subclasses).
   - Public factory functions keep their generic type parameters but
     declare their return type as a stdlib interface
     (`Iterator<T>`, `Sequence<T>`, `Iterable<T>`, `List<T>`, etc.)
     instead of the concrete internal class. The internal class still
     implements that interface, so callers get the same behavior with
     no API change visible to Kotlin or Swift consumers.
   - Tests in `commonTest` of the same module retain visibility of the
     internal classes via Kotlin's module-private rules, so
     type-specific assertions (`sizeHint`, internal `state`, `putBack`,
     etc.) keep working by constructing the internal class directly.
     Iteration-shape assertions can continue to use the public
     factory.

   **Trigger 8b — Kotlin function types in public API positions.** A
   public field, parameter, or return type of `() -> R` / `(A) -> R` /
   `(A, B) -> R` triggers the same `Unchecked cast` shape, even when
   the *enclosing* class has no generic parameter. The Kotlin source
   has no `<T>` in sight, but Kotlin function types compile to
   `Function0<R>` / `Function1<A, R>` / ... under the hood — same
   erasure surface, same untyped bridge channel. The bridge emits:

   ```kotlin
   val ____self = dereferenceExternalRCRef(self) as kotlin.Function0<kotlin.Long>
   // w: Unchecked cast of 'Any' to '() -> Long'.
   ```

   Live in-the-wild example from tree-sitter-language-kotlin CI on
   2026-05-23 (`compileSwiftExportMainKotlinMacosArm64` failed on
   `build/SwiftExport/macosArm64/Debug/files/TreeSitterLanguage/TreeSitterLanguage.kt:12:37`
   with `Unchecked cast of 'Any' to '() -> Long'.`):

   ```kotlin
   // Source (Language.kt) — no visible generic, but () -> Long erases
   class LanguageFn private constructor(private val raw: () -> Long) {
       companion object { fun fromRaw(f: () -> Long): LanguageFn = LanguageFn(f) }
       fun intoRaw(): () -> Long = raw
   }
   ```

   *Per-repo fix:* replace the public function-type with a named
   `fun interface` (single-abstract-method interface). SAM conversion
   lets every existing lambda call site stay unchanged
   (`LanguageFn.fromRaw { 42L }` still compiles), but the bridge sees a
   stable nominal type instead of an erased `FunctionN<...>`. The
   pattern:

   ```kotlin
   // Named SAM interface — bridge-friendly, lambda-friendly
   fun interface LanguageProvider {
       fun call(): Long
   }

   class LanguageFn private constructor(private val raw: LanguageProvider) {
       companion object {
           fun fromRaw(f: LanguageProvider): LanguageFn = LanguageFn(f)
       }
       fun intoRaw(): LanguageProvider = raw
   }
   ```

   `fun interface` is the right knob — not a plain `interface` (loses
   SAM conversion at call sites) and not `Function0<Long>` (same
   erasure problem in the bridge). If the function-type value is
   internal-only, marking the field `internal` is also valid.

   **Alternative when the public API must stay public.** Both triggers
   8a and 8b accept a second valid fix: tag the offending declaration
   with `@HiddenFromObjC` (and add `@file:OptIn(kotlin.experimental.ExperimentalObjCRefinement::class)`
   at the top of the file). The annotation tells the Kotlin/Native
   Obj-C / Swift bridge to skip the declaration without changing its
   Kotlin visibility, so callers in Kotlin (and in other Kotlin
   sibling repos) still see the same public surface. Use this when
   making the class `internal` would break the repo's published API
   contract or when the surface is the public face of a clean-room
   port whose Rust upstream exposed the type publicly. Evidence:
   tree-sitter-kotlin commit on `CBufferIter<T>` (trigger 8a, public
   iterator wrapper that had to stay public to mirror the upstream
   Rust API) and the follow-on sweep that batch-tagged every public
   callback typealias and enclosing class in
   `treesitter/lib/*.kt` (mix of 8a and 8b across the internal
   C-runtime subpackage). The `internal class` recipe is still the
   default when there is no API-contract reason to keep the
   declaration public; `@HiddenFromObjC` is the escape hatch when
   there is.

   **DO NOT** scope `allWarningsAsErrors=false` to the
   `compileSwiftExportMain*` task family for either trigger. That was
   tried first (itertools-kotlin PR #22 for trigger 8a) and is symptom
   suppression: it leaves the unchecked casts in the bridge and only
   mutes the compiler. The `tasks.matching { ... }.withType<KotlinCompilationTask<*>>()`
   block (and any `KotlinCompilationTask` import that accompanies it)
   should be deleted whenever found.

   With either fix in place, `allWarningsAsErrors=true` applies
   uniformly. The bridge surface shrinks accordingly — in itertools-kotlin
   it went from ~20 `@ExportedBridge` functions and 12 generic
   `@file:BindClassToObjCName(... :: class, ...)` annotations down to
   6 `@ExportedBridge` functions and zero generic class bindings,
   leaving only the genuinely-public concrete types (e.g. `Position`)
   bridged to Swift. itertools-kotlin commit `40513a5` is the canonical
   reference for trigger 8a; tree-sitter-language-kotlin (the
   `LanguageFn` / `LanguageProvider` rewrite) is the canonical reference
   for trigger 8b. tree-sitter-kotlin's batch SAM-ification of seven
   public function-type surfaces (preemptive, not reactive — done from
   the bridge-clean source before CI surfaced the warnings) is the
   canonical reference for the audit-and-sweep pattern below.

   *Audit grep (run once per repo after a fresh `embedSwiftExportForXcode`):*

   ```sh
   # Combined audit — both triggers, both shapes the bridge writes
   grep -nE "as kotlin\\.Function[0-9]+<|as io\\.github\\.kotlinmania\\.[^<]+<kotlin\\.Any" \
       build/SwiftExport/macosArm64/Debug/files/*/*.kt
   ```

   Every match in a `kotlin.FunctionN<...>` is trigger 8b (function
   type in public surface); every match in
   `io.github.kotlinmania.<pkg>.<Type><kotlin.Any` is trigger 8a
   (unconstrained generic). Both are fixable per-repo via the
   recipes below. If grep returns no results, gap #8 doesn't apply to
   this repo and no API change is needed.

## Recipe for replacing `kotlin.Result<T>` in a public API

The canonical pattern from http-kotlin commits `a179143` and the
follow-up that removed `!!`. Two principles guide the shape:

  * **The class invariant ("exactly one of value/error is non-null")
    is encoded explicitly with `init { require(...) }`.** That makes
    the invariant a real fact the type-checker can rely on after
    construction.
  * **No `!!` and no `@Suppress("UNCHECKED_CAST")`.** Both are
    workspace-forbidden per every repo's `CLAUDE.md`. `!!` is the
    runtime cousin of `@Suppress`: it suppresses the null-safety
    type-check at the cost of silencing a real bug class. The
    `when { value != null -> ...; error != null -> ...; else ->
    error("...") }` form below reads only the field whose
    non-null-ness is guaranteed in that branch, with the unreachable
    `else` as a defensive belt-and-suspenders for invariant
    violations.

```kotlin
/**
 * Result of [SomeApi.someFactoryThatCanFail].
 *
 * Flat-class shape (rather than sealed Ok / Err variants) — the
 * Kotlin Swift Export plugin does not currently emit Swift bindings
 * that let consumers pattern-match on sealed subclasses. A flat
 * class with `isSuccess()` / `isFailure()` predicates and nullable
 * `value` / `error` accessors bridges cleanly to Swift, where the
 * consumer uses `if result.isSuccess() { result.value?.thing() }`.
 *
 * The class invariant — exactly one of [value] / [error] is non-null —
 * is enforced at construction by [init]. That makes the otherwise
 * unreachable branch in [getOrThrow] expressible without `!!`.
 */
class SomeResult internal constructor(
    val value: SomeType?,
    val error: SomeConcreteError?,
) {
    init {
        require((value == null) != (error == null)) {
            "SomeResult must carry exactly one of value or error (got value=$value, error=$error)"
        }
    }

    companion object {
        internal fun ok(value: SomeType) = SomeResult(value, null)
        internal fun err(error: SomeConcreteError) = SomeResult(null, error)
    }

    fun getOrThrow(): SomeType = when {
        value != null -> value
        error != null -> throw SomeException(error)
        else -> error("SomeResult class invariant violated: both value and error are null")
    }

    fun getOrNull(): SomeType? = value

    fun isSuccess(): Boolean = value != null
    fun isFailure(): Boolean = error != null
}
```

### What NOT to do

These patterns look like they work but quietly suppress bugs:

```kotlin
// WRONG: !! suppresses the null check, hiding any future bug where
// the class invariant is violated. The `error` field "should never"
// be null when `value` is null, but the type system can't see that.
fun getOrThrow(): SomeType = value ?: throw SomeException(error!!)

// WRONG: `internal val error` hides the field from the public Swift
// surface to dodge an export problem, but Swift consumers legitimately
// need to inspect the failure. The fix is to make the Swift Export
// boundary handle the type, not to hide the API.
class SomeResult internal constructor(
    val value: SomeType?,
    internal val error: SomeConcreteError?,
)

// WRONG: sealed Ok/Err variants don't survive Swift Export — see
// upstream gap #4 above. Compiles in Kotlin but the Swift consumer
// can't do `as? SomeResultOk`.
sealed class SomeResult {
    class Ok(val value: SomeType) : SomeResult()
    class Err(val error: SomeConcreteError) : SomeResult()
}
```

---

## Recipe for hiding unconstrained generic iterator classes

The canonical pattern from itertools-kotlin commit `40513a5`. Applies
whenever a public Kotlin API exposes a generic class whose type
parameter is unconstrained (`class Foo<T>`, `class Bar<L, R>`, etc.)
and the class is reachable through the module's exported surface.

The principle: **public factories, internal implementations.** Swift
Export only emits bridge code for symbols that are part of the
public API — `internal` types are skipped. So the offending unchecked
`Any?` casts disappear if the concrete class is hidden behind a stdlib
interface return type.

### Before

```kotlin
package io.github.kotlinmania.itertools

class ZipEq<A, B>(
    private val left: Iterator<A>,
    private val right: Iterator<B>,
) : Iterator<Pair<A, B>> {
    override fun hasNext(): Boolean = TODO()
    override fun next(): Pair<A, B> = TODO()
}

fun <A, B> zipEq(left: Iterator<A>, right: Iterator<B>): ZipEq<A, B> =
    ZipEq(left, right)
```

Plugin-generated bridge file (`build/SwiftExport/.../Itertools.kt`) ends up with:

```kotlin
@ExportedBridge
fun __root___ZipEq_next(self: kotlin.native.internal.NativePtr): kotlin.native.internal.NativePtr {
    val ____self = dereferenceExternalRCRef(self) as io.github.kotlinmania.itertools.ZipEq<kotlin.Any?, kotlin.Any?>
    // ^^^ Unchecked cast: 'Any?' to 'ZipEq<Any?, Any?>'
    ...
}
```

Under `allWarningsAsErrors.set(true)` that warning becomes an error
and `compileSwiftExportMainKotlin<Target>` fails.

### After

```kotlin
package io.github.kotlinmania.itertools

internal class ZipEq<A, B>(
    private val left: Iterator<A>,
    private val right: Iterator<B>,
) : Iterator<Pair<A, B>> {
    override fun hasNext(): Boolean = TODO()
    override fun next(): Pair<A, B> = TODO()
}

fun <A, B> zipEq(left: Iterator<A>, right: Iterator<B>): Iterator<Pair<A, B>> =
    ZipEq(left, right)
```

The class is `internal`, so it never enters the Swift Export bridge.
The factory's return type changed from `ZipEq<A, B>` (the concrete
class) to `Iterator<Pair<A, B>>` (the stdlib interface). Kotlin
consumers in other modules can still call `zipEq(...)` and iterate
the result; Swift consumers get a bridge for the factory function
alone, with no `<Any?>` cast.

### Tests

Tests in `commonTest` of the same module retain visibility of
`internal` classes (Kotlin's module-private rules). When a test
asserts on a type-specific member that isn't part of `Iterator<T>` —
e.g. a custom `sizeHint`, `state`, `putBack`, internal counter —
construct the internal class directly instead of going through the
factory:

```kotlin
// commonTest can still see ZipEq's internal members
@Test
fun zipEqSizeHint() {
    val iter = ZipEq(listOf(1).iterator(), listOf("a", "b").iterator())
    assertEquals(1, iter.sizeHint())
}

// Iteration-shape assertions can still use the public factory
@Test
fun zipEqIterationShape() {
    val seq = zipEq(listOf(1, 2).iterator(), listOf("a", "b").iterator()).asSequence().toList()
    assertEquals(listOf(1 to "a", 2 to "b"), seq)
}
```

### What NOT to do

```kotlin
// WRONG: silencing the warning leaves the unchecked Any? cast in the
// bridge file. The plugin's runtime cast remains and the type-erased
// shape of the bridge stays the same — only the compiler is muted.
// Itertools-kotlin PR #22 tried this and was reverted by PR after.
tasks.matching { it.name.startsWith("compileSwiftExportMain") }
    .withType<KotlinCompilationTask<*>>()
    .configureEach {
        compilerOptions.allWarningsAsErrors.set(false)
    }

// WRONG: making the class `public abstract` and exposing only a
// public interface doesn't help — Swift Export still bridges the
// abstract class and emits the same `Any?` cast.
abstract class ZipEq<A, B> : Iterator<Pair<A, B>>

// WRONG: a `typealias` to a stdlib interface doesn't hide the
// underlying class from Swift Export. The original class still
// participates in bridging.
typealias ZipEqAlias<A, B> = ZipEq<A, B>
```

### Scope check

For each repo running Swift Export, grep the bridge file once after
a fresh `embedSwiftExportForXcode` run:

```sh
grep -nE "as io\\.github\\.kotlinmania\\.<pkg>\\.[A-Z][A-Za-z0-9_]+<kotlin\\.Any" \
    build/SwiftExport/macosArm64/Debug/files/*/*.kt
```

Every match is a candidate for the `internal class` + interface-return
treatment. If grep returns no results, this trigger (8a) doesn't apply
to this repo and no API change is needed.

---

## Recipe for hiding Kotlin function types behind a `fun interface`

The canonical pattern from the tree-sitter-language-kotlin
`LanguageFn` / `LanguageProvider` rewrite (2026-05-23). Applies
whenever a public Kotlin API uses a Kotlin function type
(`() -> R`, `(A) -> R`, `(A, B) -> R`, `suspend (A) -> R`, etc.) in a
**public** field, parameter, or return position.

The principle: **named SAM interface, not anonymous function type.**
Swift Export bridges named types as stable nominal symbols; it bridges
function types by erasing them through `Function0<R>` / `Function1<A, R>`
/ ... which collide with the `Any` channel in the bridge. A `fun interface`
gives the bridge a real symbol, and Kotlin's SAM conversion keeps lambda
call sites unchanged.

### Before

```kotlin
package io.github.kotlinmania.treesitterlanguage

class LanguageFn private constructor(private val raw: () -> Long) {
    companion object {
        fun fromRaw(f: () -> Long): LanguageFn = LanguageFn(f)
    }
    fun intoRaw(): () -> Long = raw
}
```

Plugin-generated bridge file (`build/SwiftExport/.../TreeSitterLanguage.kt`)
ends up with:

```kotlin
@ExportedBridge
fun __root___LanguageFn_intoRaw(self: kotlin.native.internal.NativePtr): kotlin.native.internal.NativePtr {
    val ____self = dereferenceExternalRCRef(self) as kotlin.Function0<kotlin.Long>
    // ^^^ Unchecked cast of 'Any' to '() -> Long'
    ...
}
```

Under `allWarningsAsErrors.set(true)` the warning becomes an error
and `compileSwiftExportMainKotlin<Target>` fails.

### After

```kotlin
package io.github.kotlinmania.treesitterlanguage

/**
 * Single-abstract-method interface that stands in for the upstream
 * Rust `unsafe extern "C" fn() -> *const c_void` function pointer.
 * Named so Swift Export can bridge a nominal symbol instead of an
 * erased `FunctionN<...>`.
 */
fun interface LanguageProvider {
    fun call(): Long
}

class LanguageFn private constructor(private val raw: LanguageProvider) {
    companion object {
        fun fromRaw(f: LanguageProvider): LanguageFn = LanguageFn(f)
    }
    fun intoRaw(): LanguageProvider = raw
}
```

`LanguageProvider` is `fun interface`, so Kotlin's SAM conversion still
accepts a lambda at the call site:

```kotlin
val fn = LanguageFn.fromRaw { tree_sitter_language() }
// Equivalent to: LanguageFn.fromRaw(LanguageProvider { tree_sitter_language() })
```

The bridge file's exported function now references `LanguageProvider`
(a nominal symbol Swift Export emits a binding for) instead of the
erased `Function0<Long>`. No `Any` cast.

### After (migration variant: name the SAM method `invoke`)

The recipe above uses `fun call()` as the SAM method, which is right
for new APIs. For repos *migrating* from a public typealias whose
call sites already use call-syntax (`callback(arg)` instead of
`callback.call(arg)`), name the SAM method `operator fun invoke(...)`
so Kotlin's invoke-operator convention keeps every existing call site
working without an edit:

```kotlin
// Before (typealias):
typealias ParseProgressCallback = (currentByteOffset: UInt, hasError: Boolean) -> Boolean

// After (fun interface with invoke):
fun interface ParseProgressCallback {
    operator fun invoke(currentByteOffset: UInt, hasError: Boolean): Boolean
}
```

Every prior caller of `callback(byte, hasError)` continues to compile
unchanged — `callback(...)` is now `callback.invoke(...)` via the
operator convention. `StableRef<ParseProgressCallback>` references in
nativeMain actuals also compile unchanged because the type name is
the same; only its kind changed (typealias → fun interface). Evidence:
tree-sitter-kotlin batch SAM-ification of seven public typealiases
and inline callback parameters, where this naming choice meant the
`jvmMain`, `androidMain`, and `nativeMain` actuals needed zero edits.

### Alternative: `@HiddenFromObjC` when the public API must stay public

The `fun interface` rewrite changes the public Kotlin API (callers
that wrote out `(UInt, Boolean) -> Boolean` explicitly must update to
the SAM type name). When the source-of-truth contract for the Kotlin
port requires keeping the original function-type shape — e.g. the
clean-room Rust upstream exposed the callback as a function-pointer
and the Kotlin port mirrors that signature publicly — tag the
declaration with `@HiddenFromObjC` instead and leave the function
type alone:

```kotlin
@file:OptIn(kotlin.experimental.ExperimentalObjCRefinement::class)

package io.github.kotlinmania.treesitter

import kotlin.native.HiddenFromObjC

@HiddenFromObjC
class CBufferIter<T> internal constructor(
    private val items: List<T>,
) : Iterator<T> {
    // ... unchanged
}
```

The annotation only affects the Kotlin/Native Obj-C / Swift bridge:
Kotlin callers (including other sibling `*-kotlin` repos) keep
seeing the same public class with the same signature. The Swift
Export bridge file skips the declaration entirely, so the
`Unchecked cast of 'Any'` line never appears. This applies to both
trigger 8a (generic class) and trigger 8b (function-type field /
parameter / return).

When most or all of a subpackage is internal-by-convention runtime
code that should never reach Swift — tree-sitter-kotlin's
`treesitter.lib` C-runtime port is the canonical example — sweep it
with file-level OptIn plus per-class `@HiddenFromObjC` rather than
tagging one declaration at a time. `@HiddenFromObjC` has no `FILE`
target, so the annotation must go on each class / property / function
individually; the file-level `@file:OptIn(ExperimentalObjCRefinement)`
is what enables those per-declaration annotations.

### Tests

SAM-convertible interfaces work the same as function types in tests
— pass a lambda or an explicit implementation, your choice:

```kotlin
@Test
fun fromRawAcceptsLambda() {
    val fn = LanguageFn.fromRaw { 42L }
    assertEquals(42L, fn.intoRaw().call())
}

@Test
fun fromRawAcceptsExplicitImpl() {
    val provider = object : LanguageProvider {
        override fun call(): Long = 99L
    }
    val fn = LanguageFn.fromRaw(provider)
    assertSame(provider, fn.intoRaw())
}
```

### What NOT to do

```kotlin
// WRONG: regular `interface` loses SAM conversion. Every existing
// `LanguageFn.fromRaw { ... }` call site breaks because Kotlin won't
// auto-wrap the lambda. Use `fun interface` instead.
interface LanguageProvider {
    fun call(): Long
}

// WRONG: switching to Function0<Long> is the same erasure problem.
// The bridge still writes `as kotlin.Function0<kotlin.Long>` because
// Function0 IS the underlying type of `() -> Long`.
class LanguageFn private constructor(private val raw: Function0<Long>)

// WRONG: silencing the warning leaves the unchecked cast in the
// bridge file. The plugin's runtime cast remains and the bridge
// shape is unchanged — only the compiler is muted.
tasks.matching { it.name.startsWith("compileSwiftExportMain") }
    .withType<KotlinCompilationTask<*>>()
    .configureEach {
        compilerOptions.allWarningsAsErrors.set(false)
    }

// WRONG: typealias to a Function type does NOT hide it from the
// bridge. The underlying function type still participates.
typealias LanguageProvider = () -> Long
```

### Scope check

For each repo running Swift Export, grep the bridge file once after
a fresh `embedSwiftExportForXcode` run:

```sh
grep -nE "as kotlin\\.Function[0-9]+<" \
    build/SwiftExport/macosArm64/Debug/files/*/*.kt
```

Every match is a candidate for the `fun interface` treatment. As a
source-side audit (preferred — it also catches surfaces the bridge
hasn't been generated for yet, e.g. new ports or new public
typealiases added to existing ports), grep for public function-type
signatures:

```sh
grep -nE "^(public |internal |private )?(class |object |val |var |fun )[^=]*: \\(.*\\) -> " \
    src/commonMain/kotlin/**/*.kt
grep -nE "^(public |internal |private )?(class |object |val |var |fun )[^=]*\\(.* \\(.*\\) -> " \
    src/commonMain/kotlin/**/*.kt
grep -rnE "^typealias [A-Z][A-Za-z0-9_]+ = \\(.*\\) -> " \
    src/commonMain/kotlin/
```

If all three passes return no results, trigger 8b doesn't apply and
no API change is needed.

### Preemptive sweep (run on every fresh port)

A new repo's public function-type surfaces should be SAM-ified
**before** Swift Export ever runs against them, not after CI surfaces
the warning. The audit greps above are cheap to run from a clean
checkout. The pattern from tree-sitter-kotlin (seven SAMs landed in
a single PR ahead of CI evidence):

1. Run the three source-side greps. Catalog every public
   function-type surface.
2. For each surface, decide:
   - **typealias whose call sites use `callback(arg)` syntax** →
     replace with `fun interface X { operator fun invoke(...) }`
     so call sites compile unchanged.
   - **new SAM, no existing call-syntax constraint** → name the SAM
     method semantically (`fun call()`, `fun read(byte: UInt)`, etc.).
   - **inline `((State) -> Boolean)?` parameter on a builder class**
     → introduce a co-located named SAM and replace the parameter
     type.
   - **subpackage that is internal-by-convention runtime port** →
     tag every class with `@HiddenFromObjC` (file-level
     `@file:OptIn(ExperimentalObjCRefinement)` enables the
     per-class annotation; there is no `@file:HiddenFromObjC` target).
3. Compile the three highest-coverage targets (`macosArm64`, `jvm`,
   `androidMain`) locally to confirm the SAM conversion didn't break
   any call site. Lambda call sites at invocation positions
   SAM-convert automatically; `StableRef<TypeName>` references stay
   the same because the type name didn't change, only its kind.

---

## Why the recipe is in every repo

Two reasons:

1. **The Claude agent operates per-repo** in the GitHub App context.
   Each repo's agent invocation only sees the repo it's working in.
   Pulling instructions from a central source would require the agent
   to make cross-repo API calls; embedding the recipe in the repo
   makes the work fully self-contained.

2. **The recipe ages with the rollout commit.** When the workarounds
   in [§ Known upstream gaps](#known-upstream-gaps) eventually become
   unnecessary, the recipe's removal (or amendment) will land in the
   same commit as the workaround removal. Future spelunkers reading
   `git log -- SWIFT_EXPORT_ROLLOUT.md` will see the full history of
   what was tried and why.

— Sydney Renee, KotlinMania
