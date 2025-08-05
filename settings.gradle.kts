rootProject.name = "kotlinrl"

include(
    "core",   //Core API
    "envs",                 //Environment Implementations
    "examples",
//    "algorithms",
//    "training",
//    "serialization",
    "integration"           //Integration tools - Gymnasium, etc.
)

//ludus – training infrastructure
//colosseum – evaluation and benchmarking
//dojo – experimental learning setups or curriculum learning
