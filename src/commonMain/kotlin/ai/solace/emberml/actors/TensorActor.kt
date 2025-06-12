package ai.solace.emberml.actors

import ai.solace.emberml.tensor.common.EmberTensor
import ai.solace.emberml.tensor.common.EmberShape
import kotlinx.coroutines.channels.SendChannel

/**
 * Base class for all tensor-related messages.
 */
sealed class TensorMessage : ActorMessage

/**
 * Request to add two tensors.
 */
data class AddTensorMessage(
    val a: EmberTensor,
    val b: EmberTensor,
    val responseChannel: SendChannel<TensorResponse>? = null
) : TensorMessage()

/**
 * Request to multiply two tensors.
 */
data class MultiplyTensorMessage(
    val a: EmberTensor,
    val b: EmberTensor,
    val responseChannel: SendChannel<TensorResponse>? = null
) : TensorMessage()

/**
 * Request to perform matrix multiplication.
 */
data class MatmulTensorMessage(
    val a: EmberTensor,
    val b: EmberTensor,
    val responseChannel: SendChannel<TensorResponse>? = null
) : TensorMessage()

/**
 * Request to reshape a tensor.
 */
data class ReshapeTensorMessage(
    val tensor: EmberTensor,
    val newShape: EmberShape,
    val responseChannel: SendChannel<TensorResponse>? = null
) : TensorMessage()

/**
 * Base class for tensor responses.
 */
sealed class TensorResponse : ActorMessage

data class TensorResultMessage(val result: EmberTensor) : TensorResponse
data class TensorErrorMessage(val error: Throwable) : TensorResponse

/**
 * Actor that handles tensor operations in a non-blocking coroutine-safe way.
 */
class TensorActor : BaseActor<TensorMessage>() {

    override suspend fun receive(message: TensorMessage) {
        when (message) {
            is AddTensorMessage -> handle(message) { it.a + it.b }
            is MultiplyTensorMessage -> handle(message) { it.a * it.b }
            is MatmulTensorMessage -> handle(message) { it.a.matmul(it.b) }
            is ReshapeTensorMessage -> handle(message) { it.tensor.reshape(it.newShape) }
        }
    }

    private suspend inline fun <reified T : TensorMessage> handle(
        message: T,
        crossinline operation: suspend (T) -> EmberTensor
    ) {
        try {
            val result = operation(message)
            message.responseChannel?.send(TensorResultMessage(result))
        } catch (e: Exception) {
            message.responseChannel?.send(TensorErrorMessage(e))
        }
    }
}