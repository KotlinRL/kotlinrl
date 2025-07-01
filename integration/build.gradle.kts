
dependencies {
    implementation(project(":core"))
    implementation("${group}:open-rl-kotlin-grpc-client:0.1.1")
    implementation("io.grpc:grpc-protobuf:1.60.0")
    implementation("io.grpc:grpc-stub:1.60.0")
    implementation("io.grpc:grpc-kotlin-stub:1.4.1")
}
