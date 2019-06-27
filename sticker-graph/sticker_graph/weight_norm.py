# Weight norm (Salimans & Kingma, 2016) implementation by Sean Morgan:
#
# https://github.com/tensorflow/tensorflow/pull/21276

from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.eager import context
from tensorflow.python.ops import nn_impl
from tensorflow.python.keras import initializers
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.engine.base_layer import InputSpec
from tensorflow.python.keras.layers import Wrapper
from tensorflow import VERSION

class WeightNorm(Wrapper):
    """ This wrapper reparameterizes a layer by decoupling the weight's
    magnitude and direction. This speeds up convergence by improving the
    conditioning of the optimization problem.
    Weight Normalization: A Simple Reparameterization to Accelerate
    Training of Deep Neural Networks: https://arxiv.org/abs/1602.07868
    Tim Salimans, Diederik P. Kingma (2016)
    WeightNorm wrapper works for keras and tf layers.

    Arguments:
      layer: a layer instance.
      data_init: If  use data dependent variable initialization
    Raises:
      ValueError: If not initialized with a  instance.
      ValueError: If  does not contain a  of weights
      NotImplementedError: If  is True and running graph execution
    """

    def __init__(self, layer, data_init=False, **kwargs):
        if not isinstance(layer, Layer):
            raise ValueError(
                'Please initialize  layer with a '
                ' instance. You passed: {input}'.format(input=layer))

        if not context.executing_eagerly() and data_init:
            raise NotImplementedError(
                'Data dependent variable initialization is not available for '
                'graph execution')

        self.initialized = True
        if data_init:
            self.initialized = False

        super(WeightNorm, self).__init__(layer, **kwargs)
        from pkg_resources import parse_version
        if parse_version(VERSION) >= parse_version("1.14"):
            self._track_trackable(layer, name='layer')
        else:
            self._track_checkpointable(layer, name='layer')
            
    def _compute_weights(self):
        """Generate weights by combining the direction of weight vector
         with it's norm """
        with variable_scope.variable_scope('compute_weights'):
            self.layer.kernel = nn_impl.l2_normalize(
                self.layer.v, axis=self.norm_axes) * self.layer.g

    def _init_norm(self, weights):
        """Set the norm of the weight vector"""
        from tensorflow.python.ops.linalg_ops import norm
        with variable_scope.variable_scope('init_norm'):
            flat = array_ops.reshape(weights, [-1, self.layer_depth])
            return array_ops.reshape(norm(flat, axis=0), (self.layer_depth,))

    def _data_dep_init(self, inputs):
        """Data dependent initialization for eager execution"""
        from tensorflow.python.ops.nn import moments
        from tensorflow.python.ops.math_ops import sqrt

        with variable_scope.variable_scope('data_dep_init'):
            # Generate data dependent init values
            activation = self.layer.activation
            self.layer.activation = None
            x_init = self.layer.call(inputs)
            m_init, v_init = moments(x_init, self.norm_axes)
            scale_init = 1. / sqrt(v_init + 1e-10)

        # Assign data dependent init values
        self.layer.g = self.layer.g * scale_init
        self.layer.bias = (-m_init * scale_init)
        self.layer.activation = activation
        self.initialized = True

    def build(self, input_shape):
        """Build """
        input_shape = tensor_shape.TensorShape(input_shape).as_list()
        self.input_spec = InputSpec(shape=input_shape)

        if not self.layer.built:
            self.layer.build(input_shape)
            self.layer.built = False

            if not hasattr(self.layer, 'kernel'):
                raise ValueError(
                    ' must wrap a layer that'
                    ' contains a  for weights'
                )

            # The kernel's filter or unit dimension is -1
            self.layer_depth = int(self.layer.kernel.shape[-1])
            self.norm_axes = list(range(self.layer.kernel.shape.ndims - 1))

            self.layer.v = self.layer.kernel
            self.layer.g = self.layer.add_variable(
                name="g",
                shape=(self.layer_depth,),
                initializer=initializers.get('ones'),
                dtype=self.layer.kernel.dtype,
                trainable=True)

            with ops.control_dependencies([self.layer.g.assign(
                    self._init_norm(self.layer.v))]):
                self._compute_weights()

            self.layer.built = True

        super(WeightNorm, self).build()
        self.built = True

    def call(self, inputs):
        """Call """
        if context.executing_eagerly():
            if not self.initialized:
                self._data_dep_init(inputs)
            self._compute_weights()  # Recompute weights for each forward pass

        output = self.layer.call(inputs)
        return output

    def compute_output_shape(self, input_shape):
        return tensor_shape.TensorShape(
            self.layer.compute_output_shape(input_shape).as_list())
