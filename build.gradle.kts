plugins {
    kotlin("multiplatform") version "2.0.21"
    kotlin("plugin.serialization") version "2.0.21"
    id("com.github.ben-manes.versions") version "0.51.0"
    id("maven-publish")
}

group = "ai.solace.emberml"
version = "0.1.0"

repositories {
    mavenCentral()
}

kotlin {
    // No JVM target - Pure native/common code only

    // Native targets
    linuxX64()
    macosX64()
    macosArm64()
    mingwX64()

    // JavaScript target
    js(IR) {
        browser()
        nodejs()
    }

    sourceSets {
        val commonMain by getting {
            dependencies {
                implementation(kotlin("stdlib"))
                implementation("org.jetbrains.kotlinx:atomicfu:0.23.1")
                implementation("org.jetbrains.kotlinx:kotlinx-coroutines-core:1.9.0")
                implementation("org.jetbrains.kotlinx:kotlinx-serialization-json:1.7.3")
            }
        }
        val commonTest by getting {
            dependencies {
                implementation(kotlin("test"))
                implementation("org.jetbrains.kotlinx:kotlinx-coroutines-test:1.9.0")
            }
        }
        // Native source sets
        val nativeMain by creating {
            dependsOn(commonMain)
            dependencies {
                // Native-specific dependencies
            }
        }
        val nativeTest by creating {
            dependsOn(commonTest)
            dependencies {
                // Native-specific test dependencies
            }
        }

        // Configure all native targets to use the native source sets
        val linuxX64Main by getting { dependsOn(nativeMain) }
        val linuxX64Test by getting { dependsOn(nativeTest) }
        val macosX64Main by getting { dependsOn(nativeMain) }
        val macosX64Test by getting { dependsOn(nativeTest) }
        val macosArm64Main by getting { dependsOn(nativeMain) }
        val macosArm64Test by getting { dependsOn(nativeTest) }
        val mingwX64Main by getting { dependsOn(nativeMain) }
        val mingwX64Test by getting { dependsOn(nativeTest) }

        // JavaScript source sets
        val jsMain by getting {
            dependencies {
                implementation("org.jetbrains.kotlinx:kotlinx-coroutines-core-js:1.9.0")
            }
        }
        val jsTest by getting {
            dependencies {
                implementation("org.jetbrains.kotlinx:kotlinx-coroutines-test-js:1.9.0")
            }
        }
    }
}

publishing {
    publications {
        withType<MavenPublication> {
            artifactId = "ember-ml-kotlin"
        }
    }
}
