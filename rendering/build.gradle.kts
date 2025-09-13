plugins {
    id("org.openjfx.javafxplugin") version "0.1.0"
}

dependencies {
    api(project(":core"))
    compileOnly("org.jetbrains.kotlinx:kotlin-jupyter-api:0.14.1-514")
}


javafx {
    version = "17.0.10"
    modules = listOf("javafx.graphics", "javafx.controls", "javafx.media")
}