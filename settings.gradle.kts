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

// Source dependency: include the sibling klang project as a composite build.
// klang declares `group = "ai.solace"` and `version = "0.7.2"`, which matches
// the `ai.solace:klang:0.7.2` coordinates in build.gradle.kts, so Gradle's
// automatic dependency substitution wires it up without a manual mapping.
includeBuild("../klang")
