import org.jetbrains.kotlin.gradle.targets.js.nodejs.NodeJsEnvSpec
import org.jetbrains.kotlin.gradle.targets.js.nodejs.NodeJsRootExtension
import org.jetbrains.kotlin.gradle.targets.js.yarn.YarnRootEnvSpec
import org.jetbrains.kotlin.gradle.targets.js.yarn.YarnRootExtension
import org.jetbrains.kotlin.gradle.targets.wasm.nodejs.WasmNodeJsEnvSpec
import org.jetbrains.kotlin.gradle.targets.wasm.yarn.WasmYarnRootEnvSpec

plugins {
    kotlin("multiplatform") version "2.3.21"
    kotlin("plugin.serialization") version "2.3.21"
    id("com.github.ben-manes.versions") version "0.54.0"
    id("maven-publish")
}

group = "ai.solace.ember"
version = "0.1.1"

repositories {
    mavenLocal()
    mavenCentral()
}

val kcoroLib = layout.projectDirectory.file("external/kcoro/lab/mirror/core/build/lib/libkcoro.a")

kotlin {
    // Native targets for Kotlin Native build
    linuxX64()
    macosArm64 {
        binaries {
            executable("poc") {
                entryPoint = "ai.solace.klang.poc.main"
            }
            executable("limbBench") {
                entryPoint = "ai.solace.ember.bench.main"
            }
            executable("heapBench") {
                entryPoint = "ai.solace.ember.bench.heapBenchMain"
            }
        }
    }
    mingwX64()

    swiftExport {
        moduleName = "EmberML"
        flattenPackage = "ai.solace.ember"
    }

    // JavaScript target (disabled until we add a Node N-API/WASM addon to supply
    // zero-copy C-layout buffers; JS GC heap is too small for parity today).
    /*
    js(IR) {
        browser()
        nodejs()
    }
    */

    sourceSets {
        val commonMain by getting {
            dependencies {
                implementation(kotlin("stdlib"))
                implementation("org.jetbrains.kotlinx:atomicfu:0.32.1")
                implementation("org.jetbrains.kotlinx:kotlinx-coroutines-core:1.11.0")
                implementation("org.jetbrains.kotlinx:kotlinx-serialization-json:1.11.0")
                implementation("ai.solace:klang:0.7.2")
            }
            kotlin.srcDir("src/commonMain/kotlin")
            resources.srcDir("src/commonMain/resources")
            kotlin.exclude(
                "ai/solace/ember/backend/metal/**",
                "ai/solace/ember/backend/storage/**",
                "ai/solace/ember/actors/**",
                "ai/solace/ember/nn/**",
                "ai/solace/ember/ops/**",
                "ai/solace/ember/tensor/**",
                "ai/solace/ember/training/**",
                "ai/solace/ember/utils/**",
                "ai/solace/ember/examples/**",
                "ai/solace/ember/Ember.kt",
                "ai/solace/emberml/**",
            )
        }
        val commonTest by getting {
            dependencies {
                implementation(kotlin("test"))
                implementation("org.jetbrains.kotlinx:kotlinx-coroutines-test:1.11.0")
            }
            kotlin.setSrcDirs(listOf("src/commonTest/kotlin/ai/solace/limbengine"))
            resources.setSrcDirs(emptyList<File>())
        }
        // Native source sets
        val nativeMain by creating {
            dependsOn(commonMain)
            dependencies {
                // Native-specific dependencies
            }
            kotlin.srcDir("src/nativeMain/kotlin")
        }
        val nativeTest by creating {
            dependsOn(commonTest)
            dependencies {
                // Native-specific test dependencies
            }
            kotlin.setSrcDirs(emptyList<File>())
        }

        // Configure all native targets to use the native source sets
        val linuxX64Main by getting {
            dependsOn(nativeMain)
        }
        val linuxX64Test by getting { dependsOn(nativeTest) }
        val macosArm64Main by getting {
            dependsOn(nativeMain)
            kotlin.setSrcDirs(emptyList<File>())
        }
        val macosArm64Test by getting { dependsOn(nativeTest) }
        val mingwX64Main by getting {
            dependsOn(nativeMain)
        }
        val mingwX64Test by getting { dependsOn(nativeTest) }

        // JavaScript source sets (JS target currently disabled in kotlin { } to prioritize native)
    }
}

// Ember currently ships with Kotlin/JS targets disabled (see kotlin { } above).
// Keep the Kotlin/JS hardening configuration guarded so it turns on automatically
// when JS/Wasm targets are re-enabled without breaking native-only builds.
(rootProject.extensions.findByName("kotlinNodeJsSpec") as? NodeJsEnvSpec)?.apply {
    version.set("22.22.2")
}

(rootProject.extensions.findByName("kotlinWasmNodeJsSpec") as? WasmNodeJsEnvSpec)?.apply {
    version.set("22.22.2")
}

(rootProject.extensions.findByName("kotlinYarnSpec") as? YarnRootEnvSpec)?.apply {
    version.set("1.22.22")
}

(rootProject.extensions.findByName("kotlinWasmYarnSpec") as? WasmYarnRootEnvSpec)?.apply {
    version.set("1.22.22")
}

(rootProject.extensions.findByName("kotlinYarn") as? YarnRootExtension)?.apply {
    resolution("diff", "8.0.3")
    resolution("**/diff", "8.0.3")
    resolution("serialize-javascript", "7.0.5")
    resolution("**/serialize-javascript", "7.0.5")
    resolution("webpack", "5.106.2")
    resolution("**/webpack", "5.106.2")
    resolution("follow-redirects", "1.16.0")
    resolution("**/follow-redirects", "1.16.0")
    resolution("lodash", "4.18.1")
    resolution("**/lodash", "4.18.1")
    resolution("ajv", "8.20.0")
    resolution("**/ajv", "8.20.0")
    resolution("brace-expansion", "5.0.5")
    resolution("**/brace-expansion", "5.0.5")
    resolution("flatted", "3.4.2")
    resolution("**/flatted", "3.4.2")
    resolution("minimatch", "10.2.5")
    resolution("**/minimatch", "10.2.5")
    resolution("picomatch", "4.0.4")
    resolution("**/picomatch", "4.0.4")
    resolution("qs", "6.15.1")
    resolution("**/qs", "6.15.1")
    resolution("socket.io-parser", "4.2.6")
    resolution("**/socket.io-parser", "4.2.6")
}

val patchedKarmaWebpackPackage =
    rootProject.layout.projectDirectory.dir("gradle/npm/karma-webpack").asFile.absolutePath.replace("\\", "/")

(rootProject.extensions.findByName("kotlinNodeJs") as? NodeJsRootExtension)?.apply {
    versions.webpack.version = "5.106.2"
    versions.webpackCli.version = "7.0.2"
    versions.karma.version = "npm:karma-maintained@6.4.7"
    versions.karmaWebpack.version = "file:$patchedKarmaWebpackPackage"
    versions.mocha.version = "12.0.0-beta-10"
    versions.kotlinWebHelpers.version = "3.1.0"
}

// Native test verbosity can be enabled ad-hoc via CLI if needed:
//   ./gradlew macosArm64Test --info --rerun-tasks \
//      -Dkotlin.tests.verbose=true

// Convenience task: run the Kotlin/Native test binary directly with a simple, verbose logger
// so per-test names and PASS/FAIL lines are printed to the console.
tasks.register<Exec>("nativeTestVerbose") {
    description = "Run macOS arm64 native tests with verbose logger"
    group = "verification"
    dependsOn("linkDebugTestMacosArm64")
    doFirst {
        val bin = layout.projectDirectory.file("build/bin/macosArm64/debugTest/test.kexe").asFile
        if (!bin.exists()) throw GradleException("Native test binary not found: $bin. Run linkDebugTestMacosArm64 first.")
        commandLine(bin.absolutePath, "--ktest_logger=SIMPLE")
    }
}

// Policy: forbid handled exceptions in tests. Fail build if tests contain try/catch or
// exception-wrapping helpers (assertFails, runCatching, etc.).
// Toggle with -PallowHandledExceptionsInTests=true to bypass, if needed temporarily.
tasks.register("forbidExceptionsInTests") {
    group = "verification"
    description = "Fails if tests contain try/catch or exception-wrapping helpers"
    doLast {
        val allow = (project.findProperty("allowHandledExceptionsInTests") as String?)?.toBoolean() == true
        if (allow) return@doLast

        // Only scan active test sources (we currently point commonTest to this subdir)
        val testRoots = listOf(
            layout.projectDirectory.dir("src/commonTest/kotlin/ai/solace/limbengine"),
        )
        val forbidden = listOf(
            "\\btry\\s*\\{",
            "\\bcatch\\s*\\(",
            "\\bassertFails\\b",
            "\\bassertFailsWith\\b",
            "\\brunCatching\\s*\\(",
            "\\bResult\\.runCatching\\s*\\(",
        ).map { Regex(it) }

        val offenders = mutableListOf<String>()
        testRoots.forEach { root ->
            if (!root.asFile.exists()) return@forEach
            root.asFile.walkTopDown()
                .filter { it.isFile && it.extension == "kt" }
                .forEach { file ->
                    val text = file.readText()
                    if (forbidden.any { it.containsMatchIn(text) }) {
                        offenders += file.relativeTo(layout.projectDirectory.asFile).path
                    }
                }
        }
        if (offenders.isNotEmpty()) {
            throw GradleException(
                "Handled exceptions are forbidden in tests. Offending files:\n" + offenders.joinToString("\n")
            )
        }
    }
}

tasks.named("macosArm64Test").configure { dependsOn("forbidExceptionsInTests") }
tasks.named("nativeTestVerbose").configure { dependsOn("forbidExceptionsInTests") }

publishing {
    publications {
        withType<MavenPublication> {
            artifactId = "ember-ml-kotlin"
        }
    }
}

// ---------------------------------------------------------------------------
// CodeQL Java/Kotlin extraction task
//
// .github/workflows/codeql.yml invokes ./gradlew codeqlCompileJvm to feed
// kotlinc-compiled commonMain through the CodeQL Java agent.
val codeqlKotlinc: Configuration by configurations.creating {
    description = "Kotlin compiler (CodeQL extraction target only — not published)"
    isCanBeResolved = true
    isCanBeConsumed = false
}

val codeqlSourceClasspath: Configuration by configurations.creating {
    description = "Runtime classpath for CodeQL extraction of commonMain sources"
    isCanBeResolved = true
    isCanBeConsumed = false
}

dependencies {
    codeqlKotlinc("org.jetbrains.kotlin:kotlin-compiler-embeddable:2.3.21")
    codeqlSourceClasspath("org.jetbrains.kotlin:kotlin-stdlib:2.3.21")
    codeqlSourceClasspath("org.jetbrains.kotlinx:kotlinx-coroutines-core-jvm:1.11.0")
    codeqlSourceClasspath("org.jetbrains.kotlinx:kotlinx-serialization-core-jvm:1.11.0")
    codeqlSourceClasspath("org.jetbrains.kotlinx:kotlinx-serialization-json-jvm:1.11.0")
    codeqlSourceClasspath("org.jetbrains.kotlinx:kotlinx-datetime-jvm:0.7.1")
    codeqlSourceClasspath("org.jetbrains.kotlinx:kotlinx-collections-immutable-jvm:0.4.0")
}

val codeqlCompileJvm = tasks.register<JavaExec>("codeqlCompileJvm") {
    description =
        "Compile commonMain Kotlin sources with kotlinc 2.3.21 for CodeQL Java/Kotlin extraction."
    group = "verification"

    classpath(codeqlKotlinc)
    mainClass.set("org.jetbrains.kotlin.cli.jvm.K2JVMCompiler")

    val outDir = layout.buildDirectory.dir("classes/kotlin/codeql-jvm")
    val sources = fileTree("src/commonMain/kotlin") { include("**/*.kt") }
    val sentinelDir = layout.buildDirectory.dir("generated/codeql-empty-source")
    inputs.files(sources).withPathSensitivity(PathSensitivity.RELATIVE)
    inputs.files(codeqlSourceClasspath).withNormalizer(ClasspathNormalizer::class.java)
    outputs.dir(outDir)
    outputs.dir(sentinelDir)

    doFirst {
        outDir.get().asFile.mkdirs()
        val sourceFiles = sources.files.toMutableList()
        if (sourceFiles.isEmpty()) {
            val sentinelFile = sentinelDir.get().asFile.resolve("io/github/kotlinmania/codeql/_CodeqlEmptySource.kt")
            sentinelFile.parentFile.mkdirs()
            sentinelFile.writeText(
                """
                // Auto-generated. Present so codeqlCompileJvm has at least
                // one Kotlin source to feed kotlinc; replaced by real
                // commonMain content once porting begins.
                package io.github.kotlinmania.codeql

                private object _CodeqlEmptySource
                """.trimIndent(),
            )
            sourceFiles += sentinelFile
        }
        args = listOf(
            "-d", outDir.get().asFile.absolutePath,
            "-classpath", codeqlSourceClasspath.asPath,
            "-jvm-target", "21",
            "-no-stdlib",
            "-no-reflect",
            "-language-version", "2.3",
            "-api-version", "2.3",
        ) + sourceFiles.map { it.absolutePath }
    }
}


tasks.register("test") {
    dependsOn("macosArm64Test")
}
