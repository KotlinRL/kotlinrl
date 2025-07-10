package io.github.kotlinrl.core.agent

interface ExperienceCallback<State, Action> {
    fun before() = { }
    fun after(experience: Experience<State, Action>) = { }
}