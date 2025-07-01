import org.jetbrains.kotlin.gradle.dsl.*
import org.jetbrains.kotlin.gradle.tasks.KotlinJvmCompile

plugins {
    kotlin("jvm") version "1.9.21"
}

repositories {
    mavenCentral()
}

allprojects {
    tasks.withType<KotlinJvmCompile>().configureEach {
        compilerOptions {
            jvmTarget.set(JvmTarget.JVM_17)
        }
    }
}
java {
    toolchain {
        languageVersion.set(JavaLanguageVersion.of(17)) // Explicitly set Java target to 17
    }
}

subprojects {
    apply(plugin = "org.jetbrains.kotlin.jvm")
    apply(plugin = "maven-publish")

    group = "org.kotlin-reinforcement-learning"
    version = "0.1.0-SNAPSHOT"

    repositories {
        mavenCentral()
        maven {
            url = uri("https://maven.pkg.github.com/KotlinRL/open-rl-kotlin-grpc-client")
            credentials {
                username = project.findProperty("gpr.user") as String? ?: System.getenv("GPR_USER")
                password = project.findProperty("gpr.key") as String? ?: System.getenv("GPR_KEY")
            }
        }
    }

    dependencies {
        implementation(kotlin("stdlib"))

        testImplementation(kotlin("test"))
        testImplementation("io.mockk:mockk:1.14.0")
        testImplementation("io.kotest:kotest-runner-junit5:5.8.0")
        testImplementation("io.kotest:kotest-assertions-core:5.8.0")
        testImplementation("io.kotest:kotest-property:5.8.0")
    }

    configure<PublishingExtension> {
        publications {
            create<MavenPublication>("mavenJava") {
                from(components["java"])
                groupId = project.group as String?
                artifactId = project.name
                version = project.version as String?
            }
        }
    }

    tasks.withType<Test> {
        useJUnitPlatform() // Enable JUnit 5
    }
    java {
        toolchain {
            languageVersion.set(JavaLanguageVersion.of(17)) // Explicitly set Java target to 17
        }
    }
}
