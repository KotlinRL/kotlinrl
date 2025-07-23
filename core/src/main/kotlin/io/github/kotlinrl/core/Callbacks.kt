package io.github.kotlinrl.core

fun <State, Action> printEpisodeStart(
    printEvery: Int
) : EpisodeCallback<State, Action> = object : EpisodeCallback<State, Action> {
    override fun onEpisodeStart(episode: Int) {
        if (episode % printEvery == 0) println("Starting episode $episode")
    }
}

fun <State, Action> printEpisodeTotalTransitions(
    printEvery: Int
) : EpisodeCallback<State, Action> = object : EpisodeCallback<State, Action> {
    override fun onEpisodeEnd(stats: EpisodeStats<State, Action>) {
        if (stats.episode % printEvery == 0)
            println("Finished episode ${stats.episode}, ${stats.trajectory.count()} transitions.")
    }
}

fun <State, Action> printEpisodeOnGoalReached(
    goalReward : Double
) : EpisodeCallback<State, Action> = object : EpisodeCallback<State, Action> {
    override fun onEpisodeEnd(stats: EpisodeStats<State, Action>) {
        if(stats.trajectory.count { it.reward == goalReward } > 0)
            println("Goal reached in episode ${stats.episode}.")
    }
}