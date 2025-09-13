plugins {
    id("org.openjfx.javafxplugin") version "0.1.0"
}

repositories {
    mavenCentral()
    maven("https://packages.jetbrains.team/maven/p/kds/kotlin-ds-maven")
}

dependencies {
    api(project(":core"))
    implementation("org.jetbrains.kotlinx:dataframe:1.0.0-dev-7089")
    api("org.jetbrains.kotlinx:kandy-lets-plot:0.8.1-dev-67")
//    implementation("org.jetbrains.kotlinx:kotlin-statistics-jvm:0.4.2-dev-2")
    compileOnly("org.jetbrains.kotlinx:kotlin-jupyter-api:0.14.1-514")
}


javafx {
    version = "17.0.10"
    modules = listOf("javafx.graphics", "javafx.controls", "javafx.media")
}