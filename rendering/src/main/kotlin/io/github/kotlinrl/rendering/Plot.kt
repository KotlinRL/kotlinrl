package io.github.kotlinrl.rendering

import org.jetbrains.kotlinx.dataframe.api.groupBy
import org.jetbrains.kotlinx.dataframe.api.map
import org.jetbrains.kotlinx.dataframe.api.toDataFrame
import org.jetbrains.kotlinx.kandy.dsl.*
import org.jetbrains.kotlinx.kandy.letsplot.feature.*
import org.jetbrains.kotlinx.kandy.letsplot.layers.*
import org.jetbrains.kotlinx.kandy.letsplot.multiplot.plotGrid
import org.jetbrains.kotlinx.kandy.letsplot.x
import org.jetbrains.kotlinx.kandy.letsplot.y
import org.jetbrains.kotlinx.kandy.util.color.Color
import org.jetbrains.kotlinx.kandy.util.context.*
import org.jetbrains.kotlinx.multik.api.math.argMax
import org.jetbrains.kotlinx.multik.ndarray.data.*
import org.jetbrains.kotlinx.multik.ndarray.operations.max
import kotlin.math.roundToInt

fun plotPolicyStateValueGrid(
    policy: D1Array<Int>,
    vTable: D1Array<Double>,
    rows: Int,
    columns: Int,
    actionSymbols: Map<Int, String>
) = run {
    require(policy.size == rows * columns) {
        "Policy length ${policy.size} doesn't match grid size $rows×$columns"
    }
    val stateDf = buildList {
        for (row in 0 until rows) {
            for (col in 0 until columns) {
                val state = row * columns + col
                val value = (vTable[state] * 100).roundToInt() / 100.0
                val action = policy[state]
                val arrow = actionSymbols[action] ?: "?"

                // Two rows per cell: one for value, one for arrow
                add(mapOf("x" to col, "y" to -row, "value" to value, "label" to "$value", "type" to "State Value Function"))
                add(mapOf("x" to col, "y" to -row, "value" to value, "label" to arrow, "type" to "Policy"))
            }
        }
    }.flatMap { it.entries }.groupBy({ it.key}, { it.value }).toDataFrame()
    plotGrid(stateDf.groupBy("type").map { (typeLabel, group) ->
        group.plot {
            layout.title = typeLabel[0]?.toString() ?: ""

            tiles {
                x("x")
                y("y")
                fillColor("value") {
                    scale = continuous(Color.BLUE..Color.WHITE)
                }
                borderLine {
                    width = 0.5
                    color = Color.BLACK
                }
            }


            text {
                x("x")
                y("y")
                label("label")
                font {
                    this.size = if (typeLabel[0] == "Policy") 18.0 else 6.0
                    color = Color.BLACK
                }
            }

            x.axis.name = "x"
            y.axis.name = "y"
        }
    })
}

fun plotPolicyActionValueGrid(
    qTable: D2Array<Double>,
    rows: Int,
    columns: Int,
    actionSymbols: Map<Int, String>
) = run {
    require(qTable.shape[0] == rows * columns) {
        "QTable length ${qTable.shape[0]} doesn't match grid size $rows×$columns"
    }
    val stateDf = buildList {
        for (row in 0 until rows) {
            for (col in 0 until columns) {
                val state = row * columns + col
                val value = ((qTable[state].max() ?: 0.0) * 100).roundToInt() / 100.0
                val action = qTable[state].argMax()
                val arrow = actionSymbols[action] ?: "?"

                // Two rows per cell: one for value, one for arrow
                add(mapOf("x" to col, "y" to -row, "value" to value, "label" to "$value", "type" to "Action Value Function"))
                add(mapOf("x" to col, "y" to -row, "value" to value, "label" to arrow, "type" to "Policy"))
            }
        }
    }.flatMap { it.entries }.groupBy({ it.key}, { it.value }).toDataFrame()
    plotGrid(stateDf.groupBy("type").map { (typeLabel, group) ->
        group.plot {
            layout.title = typeLabel[0]?.toString() ?: ""

            tiles {
                x("x")
                y("y")
                fillColor("value") {
                    scale = continuous(Color.BLUE..Color.WHITE)
                }
                borderLine {
                    width = 0.5
                    color = Color.BLACK
                }
            }


            text {
                x("x")
                y("y")
                label("label")
                font {
                    this.size = if (typeLabel[0] == "Policy") 18.0 else 6.0
                    color = Color.BLACK
                }
            }

            x.axis.name = "x"
            y.axis.name = "y"
        }
    })
}
