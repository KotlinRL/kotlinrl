package io.github.kotlinrl.envs

import io.github.kotlinrl.core.env.*
import io.github.kotlinrl.core.model.*
import io.github.kotlinrl.core.space.*
import org.jetbrains.kotlinx.multik.api.*
import org.jetbrains.kotlinx.multik.ndarray.data.*
import org.jetbrains.kotlinx.multik.ndarray.operations.*
import java.awt.*
import java.awt.image.*
import kotlin.random.*

/**
 * Represents a grid-based maze environment with the capability of simulating an agent navigating through it.
 *
 * @property exploringStarts Indicates if the agent should start in a random position. If true, the agent starts in a position
 * that is not the goal.
 * @property shapedRewards Determines if the environment uses shaped rewards. If enabled, rewards may be based on distance
 * to the goal rather than standard termination goals.
 * @property render Specifies whether the environment should render graphical output of the maze.
 * @property size Specifies the dimensions of the maze grid (size x size).
 * @property metadata A map of additional metadata about the environment.
 * @param seed Optional seed value for random number generators, ensuring reproducibility in sampling actions and observations.
 *
 * Implements the `ModelBasedEnv` interface for structured RL environments with discrete spaces.
 */
class Maze(
    val render: Boolean = true,
    override val metadata: Map<String, Any?> = emptyMap(),
    seed: Int? = null,
) : TabularModelBasedEnv {
    enum class Action(val value: Int) {
        UP(0),
        RIGHT(1),
        DOWN(2),
        LEFT(3);
    }

    private val size: Int = 5
    private val exploringStarts: Boolean = false
    private val shapedRewards: Boolean = false
    override val observationSpace = Discrete(n = size * size, start = 0, seed = seed)
    override val actionSpace = Discrete(n = 4, start = 0, seed = seed)
    override val random = seed?.let { Random(it) } ?: Random.Default
    private var state = mk.ndarray(intArrayOf(size - 1, size - 1))
    private val goal = mk.ndarray(intArrayOf(size - 1, size - 1))
    private val maze = createMaze(size * size)
    private val distances = computeDistances(size, goal.toIntArray().toList(), maze)

    /**
     * Companion object for providing utility methods to support the functionality of the Maze class.
     *
     * The methods defined in this companion object include utilities for modifying matrix values,
     * creating maze structures, and computing distances within a maze environment. These methods
     * are private and intended for internal use within the Maze class.
     */
    private companion object {
        /**
         * Updates the value at a specific row and column of the given matrix,
         * returning a new matrix with the updated value. The original matrix remains unchanged.
         *
         * @param matrix A two-dimensional list representing the matrix to update.
         * @param row The row index of the element to update.
         * @param col The column index of the element to update.
         * @param value The new value to set at the specified row and column indices.
         * @return A new matrix with the specified element updated to the given value.
         */
        fun setValueAt(
            matrix: List<List<Double>>,
            row: Int,
            col: Int,
            value: Double
        ): List<List<Double>> {
            val meshgrid = mk.meshgrid(mk.zeros(1), mk.zeros(1))
            return matrix.mapIndexed { r, rowList ->
                if (r == row) {
                    rowList.mapIndexed { c, oldValue ->
                        if (c == col) value else oldValue
                    }
                } else {
                    rowList
                }
            }
        }


        /**
         * Creates a maze represented as a map, where each key is a coordinate in the grid and its value
         * is a list of neighboring coordinates that can be reached from there.
         *
         * @param size The size of the maze, representing both the number of rows and columns (assuming a square maze).
         * @return A map where the keys are grid coordinates provided as a list of two integers (row and column),
         *         and the values are lists of adjacent coordinates representing navigable paths in the maze.
         */
        fun createMaze(size: Int): Map<List<Int>, List<List<Int>>> {
            val maze = (0 until size).flatMap { row ->
                (0 until size).map { col ->
                    listOf(row, col) to mutableListOf(
                        listOf(row - 1, col), // Above
                        listOf(row + 1, col), // Below
                        listOf(row, col - 1), // Left
                        listOf(row, col + 1)  // Right
                    ).filter { (r, c) ->
                        // Filter out coordinates that are outside the bounds of the grid
                        r in 0 until size && c in 0 until size
                    }.toMutableList()
                }
            }.toMap().toMutableMap()

            // Step 2: Define the edges of the maze
            val leftEdges = (0 until size).map { row -> mk[mk[row, 0], mk[row, -1]] }
            val rightEdges = (0 until size).map { row -> mk[mk[row, size - 1], mk[row, size]] }
            val upperEdges = (0 until size).map { col -> mk[mk[0, col], mk[-1, col]] }
            val lowerEdges = (0 until size).map { col -> mk[mk[size - 1, col], mk[size, col]] }

            // Step 3: Define the walls inside the maze
            val walls = listOf(
                listOf(listOf(1, 0), listOf(1, 1)),
                listOf(listOf(2, 0), listOf(2, 1)),
                listOf(listOf(3, 0), listOf(3, 1)),
                listOf(listOf(1, 1), listOf(1, 2)),
                listOf(listOf(2, 1), listOf(2, 2)),
                listOf(listOf(3, 1), listOf(3, 2)),
                listOf(listOf(3, 1), listOf(4, 1)),
                listOf(listOf(0, 2), listOf(1, 2)),
                listOf(listOf(1, 2), listOf(1, 3)),
                listOf(listOf(2, 2), listOf(3, 2)),
                listOf(listOf(2, 3), listOf(3, 3)),
                listOf(listOf(2, 4), listOf(3, 4)),
                listOf(listOf(4, 2), listOf(4, 3)),
                listOf(listOf(1, 3), listOf(1, 4)),
                listOf(listOf(2, 3), listOf(2, 4)),
            )

            // Step 4: Combine edges and walls into obstacles
            val obstacles = upperEdges + lowerEdges + leftEdges + rightEdges + walls

            // Step 5: Remove obstacles from the maze connectivity
            for ((src, dst) in obstacles.map { it[0] to it[1] }) {
                maze[src]?.remove(dst)
                maze[dst]?.remove(src)
            }

            return maze
        }

        /**
         * Computes the shortest distances from a specified goal position to all other positions
         * in a maze represented as an adjacency map.
         *
         * @param size The size of the maze, assumed to be square (size x size).
         * @param goal A list of two integers representing the row and column indices of the goal position in the maze.
         * @param maze A map where the keys represent positions in the maze as lists of two integers [row, column],
         *             and the values are lists of neighboring positions (also represented as lists of [row, column]).
         * @return A two-dimensional list where each element represents the shortest distance from the goal
         *         position to the position at the corresponding row and column index. Positions unreachable
         *         from the goal are marked with Double.POSITIVE_INFINITY.
         */
        fun computeDistances(
            size: Int,
            goal: List<Int>,
            maze: Map<List<Int>, List<List<Int>>>
        ): List<List<Double>> {
            var distances = List(size) { List(size) { Double.POSITIVE_INFINITY } }
            val visited = mutableSetOf<List<Int>>()
            val (startRow, startCol) = goal
            distances = setValueAt(distances, startRow, startCol, 0.0)

            while (visited != maze.keys) {
                val closest = maze.keys.filter { it !in visited }
                    .minByOrNull {
                        val (r, c) = it
                        distances[r][c]
                    }
                    ?: break

                visited.add(closest)

                for (neighbor in maze[closest] ?: emptyList()) {
                    val (row, col) = neighbor
                    val newDistance = distances[row][col] + 1
                    distances = setValueAt(distances, row, col, newDistance)
                }
            }
            return distances

        }
    }

    /**
     * Executes a single step in the environment by performing the given action.
     *
     * This method updates the current state of the environment based on the specified action,
     * computes the corresponding reward, and determines whether the episode has terminated.
     *
     * @param action The action to be executed within the environment.
     * @return A StepResult containing the new state after executing the action, the reward obtained,
     *         a flag indicating if the episode has terminated, a flag for truncation status (always false in this case),
     *         and an empty metadata map.
     */
    override fun step(action: Int): StepResult<Int> {
        val reward = computeReward(state)
        state = nextState(state, action)
        val terminated = state == goal
        return StepResult(state[0] * state[1], reward, terminated, false, emptyMap())
    }

    /**
     * Resets the environment to its initial state and optionally allows configuration
     * through a random seed or additional options.
     *
     * If `exploringStarts` is enabled, the initial state will be sampled until it does not
     * match the `goal` state. Otherwise, the state defaults to a predefined starting position.
     *
     * @param seed An optional integer seed for the random number generator, ensuring reproducibility
     *             of the initial state if provided.
     * @param options A map of optional configuration parameters for the environment reset process.
     *                These options can include additional customization for the reset behavior.
     * @return An `InitialState` object containing the reset initial state of the environment along
     *         with any associated metadata.
     */
    override fun reset(seed: Int?, options: Map<String, Any?>?): InitialState<Int> {
        if (exploringStarts) {
            while (state[0] == goal[0] && state[1] == goal[1]) {
                val sample = observationSpace.sample()
                state = mk.ndarray(intArrayOf(sample / size, sample % size))
            }
        } else {
            state = mk.ndarray(intArrayOf(0, 0))
        }
        return InitialState(state[0] * state[1])
    }

    /**
     * Renders the current state of the maze environment as a visual representation.
     *
     * This method generates an image depicting the maze, including elements such as walls,
     * the agent's position, and the goal. The rendering process produces a byte array
     * representation of the image, suitable for visualization or further processing.
     *
     * @return A `Rendering` object, either:
     * - `Rendering.Empty` if rendering is disabled.
     * - `Rendering.RenderFrame` containing the rendered frame's dimensions and byte data.
     */
    override fun render(): Rendering {
        if (!render) return Rendering.Empty

        val cellSize = 80
        val imgSize = size * cellSize
        val image = BufferedImage(imgSize, imgSize, BufferedImage.TYPE_3BYTE_BGR)
        val g: Graphics2D = image.createGraphics()

        // Background
        g.color = Color(22, 36, 71)
        g.fillRect(0, 0, imgSize, imgSize)

        // Interior Walls – thin
        g.color = Color.WHITE
        g.stroke = BasicStroke(1f)
        for (row in 0 until size) {
            for (col in 0 until size) {
                val state = listOf(row, col)
                val neighbors = maze[state].orEmpty().toSet()

                val x = col * cellSize
                val y = row * cellSize

                if (listOf(row - 1, col) !in neighbors) g.drawLine(x, y, x + cellSize, y)                       // Top
                if (listOf(row + 1, col) !in neighbors) g.drawLine(
                    x,
                    y + cellSize,
                    x + cellSize,
                    y + cellSize
                ) // Bottom
                if (listOf(row, col - 1) !in neighbors) g.drawLine(x, y, x, y + cellSize)                       // Left
                if (listOf(row, col + 1) !in neighbors) g.drawLine(x + cellSize, y, x + cellSize, y + cellSize) // Right
            }
        }

        // Border Walls – thick
        g.stroke = BasicStroke(3f)
        g.drawLine(0, 0, imgSize, 0)                      // Top
        g.drawLine(0, imgSize - 1, imgSize, imgSize - 1)  // Bottom
        g.drawLine(0, 0, 0, imgSize)                      // Left
        g.drawLine(imgSize - 1, 0, imgSize - 1, imgSize)  // Right

        // Goal
        g.color = Color(40, 199, 172)
        g.fillRect(goal[1] * cellSize + 10, goal[0] * cellSize + 10, cellSize - 20, cellSize - 20)

        // Agent
        g.color = Color(228, 63, 90)
        val cx = state[1] * cellSize + cellSize / 2
        val cy = state[0] * cellSize + cellSize / 2
        val r = (cellSize * 0.6 / 2).toInt()
        g.fillOval(cx - r, cy - r, r * 2, r * 2)

        g.dispose()

        // Convert to RGB byte array
        val pixels = IntArray(imgSize * imgSize)
        image.getRGB(0, 0, imgSize, imgSize, pixels, 0, imgSize)

        val rgbBytes = ByteArray(pixels.size * 3)
        for (i in pixels.indices) {
            val pixel = pixels[i]
            rgbBytes[i * 3] = ((pixel shr 16) and 0xFF).toByte() // R
            rgbBytes[i * 3 + 1] = ((pixel shr 8) and 0xFF).toByte()  // G
            rgbBytes[i * 3 + 2] = (pixel and 0xFF).toByte()          // B
        }

        return Rendering.RenderFrame(width = imgSize, height = imgSize, bytes = rgbBytes.copyOf())
    }

    /**
     * Closes the maze environment and performs any necessary cleanup operations.
     *
     * This method is typically used to release resources associated with the
     * environment, ensuring proper termination and avoiding potential resource leaks.
     * After invoking this method, the environment should no longer be used.
     */
    override fun close() {

    }

    /**
     * Converts the current maze environment into a `TabularMDP` representation.
     *
     * This method constructs an MDP by defining the states, actions, reward function,
     * transition probabilities, and a discount factor for the maze environment.
     *
     * @param gamma The discount factor for the MDP, used to balance immediate and future rewards.
     *              It should be a value in the range [0, 1], where 0 focuses only on immediate rewards,
     *              and 1 considers future rewards fully.
     * @return A `TabularMDP` representation of the maze environment,
     *         containing states, actions, rewards, transitions, and the discount factor.
     */
    override fun asMDP(gamma: Double): TabularMDP {
        return TabularMDP(
            S = FiniteStates(size * size),
            A = FixedIntActions(4),
            RA = mk.d2array(size * size, 4) { s ->
                computeReward(mk.ndarray(intArrayOf(s / size, s % size)))
            },
            TA = mk.d3arrayIndices<Double>(size * size, 4, size * size) { s, a, sP ->
                val state =  mk.ndarray(intArrayOf(s / size, s % size))
                val nextState = mk.ndarray(intArrayOf(sP / size, sP % size))
                if(nextState(state, a) == nextState) 1.0 else 0.0
            },
            gamma = gamma,
        )
    }

    /**
     * Determines the next state of the agent based on the current state and the specified action.
     *
     * The method computes the potential next state by applying the chosen action.
     * If the resulting state is valid (i.e., the action keeps the agent within the constraints
     * of the maze), it transitions to that state. Otherwise, the agent remains in the current state.
     *
     * @param state The current state represented as a 1D NDArray containing the row and column indices.
     * @param action The action to be performed, represented as an integer (0: UP, 1: RIGHT, 2: DOWN, 3: LEFT).
     * @return The next state as a 1D NDArray of the same format as the input state. If the action
     *         results in an invalid transition, the current state is returned.
     */
    private fun nextState(state: NDArray<Int, D1>, action: Int): NDArray<Int, D1> {
        val (row, col) = state.toList()
        val nextState = when (action) {
            0 -> mk.ndarray(intArrayOf(row - 1, col)) // Move UP
            1 -> mk.ndarray(intArrayOf(row, col + 1)) // Move RIGHT
            2 -> mk.ndarray(intArrayOf(row + 1, col)) // Move DOWN
            3 -> mk.ndarray(intArrayOf(row, col - 1)) // Move LEFT
            else -> error("Action value not supported: $action")
        }

        return if (maze[mk[row, col]]?.contains(nextState.toList()) == true) {
            nextState
        } else {
            state
        }
    }

    /**
     * Computes the reward for the given state in the environment.
     *
     * The reward computation depends on whether shaped rewards are enabled.
     * If shaped rewards are enabled, the reward is based on the normalized
     * distance from the goal. Otherwise, a fixed reward is given for
     * reaching the goal, and a penalty is applied otherwise.
     *
     * @param state The current state represented as a 1D NDArray containing the row and column indices.
     * @return A Double representing the computed reward for the given state.
     */
    private fun computeReward(state: NDArray<Int, D1>): Double {
        return if (shapedRewards) {
            val goalDistance = distances[state[0]][state[1]]
            val maxDistance = distances.flatten().maxOrNull() ?: Double.POSITIVE_INFINITY
            -(goalDistance / maxDistance)
        } else {
            if (state == goal) 0.0 else -1.0
        }
    }
}