package io.github.kotlinrl.core

fun <State, Action> printEpisodeStart(
    printEvery: Int
) : EpisodeCallback<State, Action> = object : EpisodeCallback<State, Action> {
    override fun onEpisodeStart(episode: Int) {
        if (episode % printEvery == 0) println("Starting episode $episode")
    }
}