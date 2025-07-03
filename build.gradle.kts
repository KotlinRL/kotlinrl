import org.jetbrains.kotlin.gradle.dsl.*
import org.jetbrains.kotlin.gradle.tasks.KotlinJvmCompile

plugins {
    kotlin("jvm") version "1.9.21"
    id("com.vanniktech.maven.publish") version "0.33.0"
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
    group = "io.github.kotlinrl"
    version = "0.1.0-SNAPSHOT"

    repositories {
        mavenCentral()
        maven {
            name = "Central Portal Snapshots"
            url = uri("https://central.sonatype.com/repository/maven-snapshots/")
        }
    }
}

subprojects {
    apply(plugin = "org.jetbrains.kotlin.jvm")
    apply(plugin = "com.vanniktech.maven.publish")

    dependencies {
        implementation(kotlin("stdlib"))

        testImplementation(kotlin("test"))
        testImplementation("io.mockk:mockk:1.14.0")
        testImplementation("io.kotest:kotest-runner-junit5:5.8.0")
        testImplementation("io.kotest:kotest-assertions-core:5.8.0")
        testImplementation("io.kotest:kotest-property:5.8.0")
    }

    tasks.withType<Test> {
        useJUnitPlatform() // Enable JUnit 5
    }

    java {
        toolchain {
            languageVersion.set(JavaLanguageVersion.of(17)) // Explicitly set Java target to 17
        }
    }
    mavenPublishing {
        publishToMavenCentral()

        signAllPublications()

        coordinates(group.toString(), project.name, version.toString())

        pom {
            name.set("Kotlin Reinforcement Learning")
            description.set(project.description)
            url.set("https://github.com/KotlinRL/kotlinrl")
            licenses {
                license {
                    name.set("Apache License 2.0")
                    url.set("https://www.apache.org/licenses/LICENSE-2.0.txt")
                }
            }
            developers {
                developer {
                    id.set("dkrieg")
                    name.set("Daniel Krieg")
                    email.set("daniel_krieg@mac.com")
                }
            }
            scm {
                connection.set("scm:git:https://github.com/KotlinRL/kotlinrl.git")
                developerConnection.set("scm:git:ssh://github.com:KotlinRL/kotlinrl.git")
                url.set("https://github.com/KotlinRL/kotlinrl")
            }
        }
    }
}
