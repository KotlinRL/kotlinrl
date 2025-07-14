package io.github.kotlinrl.core

typealias EpisodeTrainer<State, Action> = io.github.kotlinrl.core.train.EpisodeTrainer<State, Action>
typealias EpisodeCallback<State, Action> = io.github.kotlinrl.core.train.EpisodeCallback<State, Action>
typealias Trainer = io.github.kotlinrl.core.train.Trainer
typealias TrainingResult = io.github.kotlinrl.core.train.TrainingResult
typealias Env<State, Action, ObservationSpace, ActionSpace> = io.github.kotlinrl.core.env.Env<State, Action, ObservationSpace, ActionSpace>

fun <State, Action> episodicTrainer(
    env: Env<State, Action, *, *>,
    agent: Agent<State, Action>,
    maxStepsPerEpisode: Int = 10_000,
    callbacks: List<EpisodeCallback<State, Action>> = emptyList()
): Trainer = EpisodeTrainer(
    env = env,
    agent = agent,
    maxStepsPerEpisode = maxStepsPerEpisode,
    callbacks = callbacks
)
