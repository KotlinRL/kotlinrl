plugins {
    id("org.openjfx.javafxplugin") version "0.1.0"
}

project.description = "Core API of KotlinRL"

dependencies {
    api("org.jetbrains.kotlinx:multik-core:0.2.0")
    api("org.jetbrains.kotlinx:multik-default:0.2.0")
    implementation("org.jcodec:jcodec:0.2.5")
    implementation("org.jcodec:jcodec-javase:0.2.5")
    compileOnly("org.jetbrains.kotlinx:kotlin-jupyter-api:0.14.1-514")
}

javafx {
    version = "24.0.1"
    modules = listOf("javafx.media")
}