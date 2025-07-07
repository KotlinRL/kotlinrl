project.description = "Integration of KotlinRL with open-rl-env"

dependencies {
    api(project(":core"))
    api("${group}:open-rl-kotlin-grpc-client:0.1.0-SNAPSHOT")
    api("io.grpc:grpc-protobuf:1.60.0")
    api("io.grpc:grpc-stub:1.60.0")
    api("io.grpc:grpc-kotlin-stub:1.4.1")
}
