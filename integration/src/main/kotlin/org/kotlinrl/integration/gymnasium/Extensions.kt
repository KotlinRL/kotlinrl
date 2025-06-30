package org.kotlinrl.integration.gymnasium

import com.google.protobuf.*
import open.rl.env.*
import open.rl.env.EnvOuterClass.RenderResponse.FrameCase.*
import org.jetbrains.kotlinx.multik.api.*
import org.jetbrains.kotlinx.multik.ndarray.data.*
import org.kotlinrl.core.env.*
import org.kotlinrl.core.env.Rendering.Empty
import org.kotlinrl.core.env.Rendering.RenderFrame
import org.kotlinrl.core.env.Rendering.Text
import org.kotlinrl.core.space.*
import org.kotlinrl.open.env.*

fun EnvOuterClass.Space.toBoxNDArrayD1Float(seed: Int?): BoxNDArrayD1<Float> {
    return BoxNDArrayD1(
        low = this.box.low.toNDArrayFloat1(),
        high = this.box.high.toNDArrayFloat1(),
        type = Float::class.java,
        seed = seed,
    )
}

fun EnvOuterClass.Space.toDiscrete(seed: Int?): Discrete {
    return Discrete(
        n = this.discrete.n,
        start = this.discrete.start,
        seed = seed
    )
}

fun EnvOuterClass.NDArray.toNDArrayFloat1(): NDArray<Float, D1> {
    return mk.ndarray(this.data.toFloatArray())
}

fun Struct.toMap(): Map<String, Any> {
    return this.fieldsMap.mapValues { (_, v) -> v.stringValue }
}

fun NDArray<Float, D1>.toNDArray(): EnvOuterClass.NDArray {
    return EnvOuterClass.NDArray.newBuilder()
        .setDtype(this.dtype.toDType())
        .addAllShape(this.shape.toList())
        .setData(this.data.getByteArray().toByteString())
        .build()
}

fun DataType.toDType(): EnvOuterClass.DType {
    return when (this) {
        DataType.ByteDataType -> EnvOuterClass.DType.uint8
        DataType.IntDataType -> EnvOuterClass.DType.int32
        DataType.LongDataType -> EnvOuterClass.DType.int64
        DataType.FloatDataType -> EnvOuterClass.DType.float32
        DataType.DoubleDataType -> EnvOuterClass.DType.float64
        else -> error("Unsupported data type: $this")
    }
}

fun ByteArray.toByteString(): ByteString = ByteString.copyFrom(this)

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
        ANSI -> Text(this.ansi)
        else -> Empty
    }
}
