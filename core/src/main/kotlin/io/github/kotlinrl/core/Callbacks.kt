package io.github.kotlinrl.core

fun <State, Action> printEpisodeStart(
    printEvery: Int
): EpisodeCallback<State, Action> = onEpisodeStart { episode ->
    if (episode % printEvery == 0) println("Starting episode $episode")
}

fun <State, Action> printEpisodeTotalTransitions(
    printEvery: Int
): EpisodeCallback<State, Action> = onEpisodeEnd {
    if (it.lastEpisode % printEvery == 0)
        println("Finished episode ${it.lastEpisode}, ${it.lastEpisodeSteps} transitions.")
}

fun <State, Action> printEpisodeOnGoalReached(printEvery: Int = 1): EpisodeCallback<State, Action> = onEpisodeEnd {
    if (it.lastEpisodeReachedGoal && it.lastEpisode % printEvery == 0)
        println("Goal reached in episode ${it.lastEpisode}.")
}

fun <State, Action> onEpisodeStart(
    f: (episode: Int) -> Unit
): EpisodeCallback<State, Action> = object : EpisodeCallback<State, Action> {
    override fun onEpisodeStart(episode: Int) = f(episode)
}

fun <State, Action> onEpisodeEnd(
    block: (result: TrainingResult) -> Unit
): EpisodeCallback<State, Action> = object : EpisodeCallback<State, Action> {
    override fun onEpisodeEnd(result: TrainingResult) = block(result)
}