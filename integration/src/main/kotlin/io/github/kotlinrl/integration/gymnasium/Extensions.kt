package io.github.kotlinrl.integration.gymnasium

import com.google.protobuf.*
import io.github.kotlinrl.core.env.*
import io.github.kotlinrl.core.env.Rendering.*
import io.github.kotlinrl.core.space.*
import io.github.kotlinrl.open.env.*
import open.rl.env.*
import open.rl.env.EnvOuterClass.DType.*
import open.rl.env.EnvOuterClass.Observation.ValueCase.*
import open.rl.env.EnvOuterClass.RenderResponse.FrameCase.*
import org.jetbrains.kotlinx.multik.api.*
import org.jetbrains.kotlinx.multik.ndarray.data.*
import org.jetbrains.kotlinx.multik.ndarray.data.DataType.*
import java.nio.*

fun EnvOuterClass.Space.toTypedSpace(seed: Int?): Space<*> {
    return when {
        hasDiscrete() -> this.toDiscrete(seed)
        hasBox() -> this.toBox(seed)
        hasTuple() -> this.toTuple(seed)
        else -> TODO()
    }
}

fun Map<String, Any?>.toStruct(): Struct {
    fun anyToValue(key: String, value: Any?): Value {
        return when (value) {
            null -> Value.newBuilder().setNullValueValue(0).build()
            is String -> Value.newBuilder().setStringValue(value).build()
            is Number -> Value.newBuilder().setNumberValue(value.toDouble()).build()
            is Boolean -> Value.newBuilder().setBoolValue(value).build()
            is Map<*, *> -> {
                @Suppress("UNCHECKED_CAST")
                val nestedMap = value as Map<String, Any?>
                Value.newBuilder().setStructValue(nestedMap.toStruct()).build()
            }
            is List<*> -> {
                val listValues = value.mapIndexed { i, v -> anyToValue("$key[$i]", v) }
                Value.newBuilder().setListValue(ListValue.newBuilder().addAllValues(listValues)).build()
            }

            else -> throw IllegalArgumentException("Unsupported type for Struct conversion: ${value.let { it::class }} at key: $key")
        }
    }

    val structBuilder = Struct.newBuilder()
    for ((key, value) in this) {
        structBuilder.putFields(key, anyToValue(key, value))
    }
    return structBuilder.build()
}

fun Struct.toMap(): Map<String, Any> {
    return this.fieldsMap.mapValues { (_, v) -> v.stringValue }
}

@Suppress("UNCHECKED_CAST")
fun <State> EnvOuterClass.Observation.toTypedState():State {
    return when (this.valueCase) {
        INT32 -> this.int32
        DOUBLE -> this.double
        STRING -> this.string
        ARRAY -> this.array.toNDArray()
        TUPLE -> this.tuple.itemsList.map { it.toTypedState() as State }.toList()
        MAP -> TODO()
        VALUE_NOT_SET -> error("State value not set.")
    } as State
}

fun EnvOuterClass.RenderResponse.toRendering(): Rendering {
    return when (this.frameCase) {
        RGB_ARRAY -> {
            val (height, width) = this.rgbArray.shapeList
            RenderFrame(
                width = width,
                height = height,
                bytes = this.rgbArray.data.toByteArray()
            )
        }
        else -> Rendering.Empty
    }
}

private fun EnvOuterClass.Space.toDiscrete(seed: Int?): Discrete {
    return Discrete(
        n = this.discrete.n,
        start = this.discrete.start,
        seed = seed
    )
}

private fun EnvOuterClass.Space.toBox(seed: Int?): Box<*, *> {
    return when (this.box.dtype) {
        float32 -> Box(
            low = mk.ndarray(
                elements = this.box.low.data.toFloatArray().toList(),
                shape = this.box.shapeList.toIntArray(),
                dim = dimensionOf(this.box.shapeCount)
            ),
            high = mk.ndarray(
                elements = this.box.high.data.toFloatArray().toList(),
                shape = this.box.shapeList.toIntArray(),
                dim = dimensionOf(this.box.shapeCount)
            ),
            dtype = FloatDataType,
            seed = seed
        )

        float64 -> Box(
            low = mk.ndarray(
                elements = this.box.low.data.toDoubleArray().toList(),
                shape = this.box.shapeList.toIntArray(),
                dim = dimensionOf(this.box.shapeCount)
            ),
            high = mk.ndarray(
                elements = this.box.high.data.toDoubleArray().toList(),
                shape = this.box.shapeList.toIntArray(),
                dim = dimensionOf(this.box.shapeCount)
            ),
            dtype = DoubleDataType,
            seed = seed
        )

        int32 -> Box(
            low = mk.ndarray(
                elements = this.box.low.data.toIntArray().toList(),
                shape = this.box.shapeList.toIntArray(),
                dim = dimensionOf(this.box.shapeCount)
            ),
            high = mk.ndarray(
                elements = this.box.high.data.toIntArray().toList(),
                shape = this.box.shapeList.toIntArray(),
                dim = dimensionOf(this.box.shapeCount)
            ),
            dtype = IntDataType,
            seed = seed
        )

        int64 -> Box(
            low = mk.ndarray(
                elements = this.box.low.data.toLongArray().toList(),
                shape = this.box.shapeList.toIntArray(),
                dim = dimensionOf(this.box.shapeCount)
            ),
            high = mk.ndarray(
                elements = this.box.high.data.toLongArray().toList(),
                shape = this.box.shapeList.toIntArray(),
                dim = dimensionOf(this.box.shapeCount)
            ),
            dtype = LongDataType,
            seed = seed
        )

        bool, uint8 -> Box(
            low = mk.ndarray(
                elements = this.box.low.data.toByteArray().toList(),
                shape = this.box.shapeList.toIntArray(),
                dim = dimensionOf(this.box.shapeCount)
            ),
            high = mk.ndarray(
                elements = this.box.high.data.toByteArray().toList(),
                shape = this.box.shapeList.toIntArray(),
                dim = dimensionOf(this.box.shapeCount)
            ),
            dtype = ByteDataType,
            seed = seed
        )

        UNRECOGNIZED -> error("Unknown dtype: ${this.box.dtype.name}")
    }
}

@Suppress("UNCHECKED_CAST")
private fun EnvOuterClass.Space.toTuple(seed: Int?): Tuple {
    return Tuple(this.tuple.spacesList.map { it.toTypedSpace(seed) }.toList() as List<Space<Any>>, seed)
}

private fun ByteString.toIntArray(): IntArray {
    val bb = ByteBuffer.wrap(this.toByteArray()).order(ByteOrder.LITTLE_ENDIAN)
    val arr = IntArray(this.size() / 4)
    for (i in arr.indices) {
        arr[i] = bb.int
    }
    return arr
}

private fun ByteString.toLongArray(): LongArray {
    val bb = ByteBuffer.wrap(this.toByteArray()).order(ByteOrder.LITTLE_ENDIAN)
    val arr = LongArray(this.size() / 8)
    for (i in arr.indices) {
        arr[i] = bb.long
    }
    return arr
}

private fun EnvOuterClass.NDArray.toNDArray(): NDArray<*, *> {
    return when (this.dtype) {
        float32 -> mk.ndarray(
            elements = this.data.toFloatArray().toList(),
            shape = this.shapeList.toIntArray(),
            dim = dimensionOf(this.shapeCount)
        )

        float64 -> mk.ndarray(
            elements = this.data.toDoubleArray().toList(),
            shape = this.shapeList.toIntArray(),
            dim = dimensionOf(this.shapeCount)
        )

        int32 -> mk.ndarray(
            elements = this.data.toIntArray().toList(),
            shape = this.shapeList.toIntArray(),
            dim = dimensionOf(this.shapeCount)
        )

        int64 -> mk.ndarray(
            elements = this.data.toLongArray().toList(),
            shape = this.shapeList.toIntArray(),
            dim = dimensionOf(this.shapeCount)
        )

        bool, uint8 -> mk.ndarray(
            elements = this.data.toByteArray().toList(),
            shape = this.shapeList.toIntArray(),
            dim = dimensionOf(this.shapeCount)
        )

        UNRECOGNIZED -> error("Invalid data type: ${this.dtype}")
    } as NDArray<*, *>
}

fun IntArray.toByteArray(): ByteArray {
    val bb = ByteBuffer.allocate(this.size * 4).order(ByteOrder.LITTLE_ENDIAN)
    for (i in this.indices) {
        bb.putInt(this[i])
    }
    return bb.array()
}
fun LongArray.toByteArray(): ByteArray {
    val bb = ByteBuffer.allocate(this.size * 8).order(ByteOrder.LITTLE_ENDIAN)
    for (i in this.indices) {
        bb.putLong(this[i])
    }
    return bb.array()
}
fun FloatArray.toByteArray(): ByteArray {
    val bb = ByteBuffer.allocate(this.size * 4).order(ByteOrder.LITTLE_ENDIAN)
    for (i in this.indices) {
        bb.putFloat(this[i])
    }
    return bb.array()
}
fun DoubleArray.toByteArray(): ByteArray {
    val bb = ByteBuffer.allocate(this.size * 8).order(ByteOrder.LITTLE_ENDIAN)
    for (i in this.indices) {
        bb.putDouble(this[i])
    }
    return bb.array()
}