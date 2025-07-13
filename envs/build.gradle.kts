plugins {
    id("org.openjfx.javafxplugin") version "0.1.0"
}


dependencies {
    api(project(":core"))
}

javafx {
    version = "24.0.1"
    modules = listOf("javafx.media")
}