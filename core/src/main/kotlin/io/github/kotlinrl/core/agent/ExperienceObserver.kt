package io.github.kotlinrl.core.agent

fun interface ExperienceObserver<State, Action> {
    operator fun invoke(experience: Experience<State, Action>)
}