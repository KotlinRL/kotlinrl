rootProject.name = "kotlinrl"

include(
    "core",   //Core API
    "tabular",
    "envs",                 //Environment Implementations
//    "examples",
//    "deep",
//    "rendering",
    "integration"           //Integration tools - Gymnasium, etc.
)

//ludus – training infrastructure
//colosseum – evaluation and benchmarking
//dojo – experimental learning setups or curriculum learning
