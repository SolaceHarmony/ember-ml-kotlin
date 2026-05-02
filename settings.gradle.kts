rootProject.name = "ember-ml-kotlin"

pluginManagement {
    repositories {
        gradlePluginPortal()
        mavenCentral()
    }
}

plugins {
    id("org.gradle.toolchains.foojay-resolver-convention") version("0.9.0")
}

// Source dependency: include the vendored klang fork at external/klang as a
// composite build. The vendored fork declares `group = "ai.solace"` and
// `version = "0.7.2"`, which matches the `ai.solace:klang:0.7.2` coordinates
// in build.gradle.kts, so Gradle's automatic dependency substitution wires it
// up without a manual mapping. The sibling `../klang` republishes under
// `io.github.kotlinmania:klang` and has dropped APIs (CDouble,
// copyFromIntArray, copyToIntArray) that this project still uses, so
// substituting against it is not viable.
includeBuild("external/klang")
