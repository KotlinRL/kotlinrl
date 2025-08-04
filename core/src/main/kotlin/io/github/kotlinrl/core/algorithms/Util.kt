package io.github.kotlinrl.core.algorithms

import org.jetbrains.kotlinx.multik.api.*
import org.jetbrains.kotlinx.multik.ndarray.data.*
import org.jetbrains.kotlinx.multik.ndarray.operations.*


data class StateActionKey<State : Comparable<State>, Action : Comparable<Action>>(
    val state: State,
    val action: Action
) : Comparable<StateActionKey<State, Action>> {
    override fun compareTo(other: StateActionKey<State, Action>): Int {
        val stateCmp = state.compareTo(other.state)
        return if (stateCmp != 0) stateCmp else action.compareTo(other.action)
    }
}

@JvmInline
value class ComparableIntList(val data: List<Int>) : Comparable<ComparableIntList> {
    override fun compareTo(other: ComparableIntList): Int {
        return data.zip(other.data).fold(0) { acc, (a, b) ->
            if (acc != 0) acc else a.compareTo(b)
        }
    }

    fun toNDArray(): NDArray<Int, DN> = mk.ndarray(data).asDNArray()

    override fun toString() = data.joinToString(",")
}


@Suppress("UNCHECKED_CAST")
internal fun <State, Action> stateActionKey(s: State, a: Action): StateActionKey<*, *> =
    when (s) {
        is NDArray<*, *> if a is Int -> StateActionKey(ComparableIntList((s as NDArray<Int, DN>).toList()), a)
        is Comparable<*> if a is Comparable<*> -> {
            StateActionKey(s as Comparable<Any>, a as Comparable<Any>)
        }

        else -> error("State ($s) and Action ($a) must be Comparable or mappable to a comparable form.")
    }

@Suppress("UNCHECKED_CAST")
internal fun <State> stateKey(s: State): Comparable<*> =
    when (s) {
        is NDArray<*, *> -> ComparableIntList((s as NDArray<Int, DN>).toList())
        is Comparable<*> -> s
        else -> error("State $s must be Comparable or mappable to a comparable key.")
    }
