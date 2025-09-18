package io.github.kotlinrl.core.env

import io.github.kotlinrl.core.*

/**
 * Represents a tabular environment where both the state and action spaces are discrete.
 * This interface extends the generic `Env` interface, using `Int` as the type for both
 * states and actions, and `Discrete` as the type for both observation and action spaces.
 *
 * It is designed for environments that operate in fully enumerable spaces, making it
 * suitable for problems like grid-worlds or board games where states and actions can
 * be mapped to integers.
 */
interface TabularEnv : Env<Int, Int, Discrete, Discrete> {
}