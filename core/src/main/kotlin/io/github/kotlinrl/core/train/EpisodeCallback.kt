package io.github.kotlinrl.core.train

interface EpisodeCallback<State, Action> {
    fun onEpisodeStart(episode: Int) {}

    fun onEpisodeEnd(result: TrainingResult) {}
}