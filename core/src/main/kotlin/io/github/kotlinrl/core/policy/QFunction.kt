package io.github.kotlinrl.core.policy

interface QFunction<State, Action> {
    operator fun get(state: State, action: Action): Double

    fun update(state: State, action: Action, value: Double): QFunction<State, Action>

    fun maxValue(state: State): Double

    fun bestAction(state: State): Action
}

/*
    fun save(path: String)

    fun load(path: String)

 */