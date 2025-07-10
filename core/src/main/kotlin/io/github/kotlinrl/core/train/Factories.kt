package io.github.kotlinrl.core.train

import io.github.kotlinrl.core.agent.*
import io.github.kotlinrl.core.env.*

fun <State, Action> trainer(
    env: Env<State, Action, *, *>,
    agent: Agent<State, Action>,
    maxStepsPerEpisode: Int = 10_000,
    callbacks: List<EpisodeCallback<State, Action>> = emptyList()
): Trainer = BasicTrainer(
    env = env,
    agent = agent,
    maxStepsPerEpisode = maxStepsPerEpisode,
    callbacks = callbacks
)