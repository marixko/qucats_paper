import tensorflow as tf
import tensorflow_probability as tfp; tfd = tfp.distributions


Input = tf.keras.layers.Input
Dense = tf.keras.layers.Dense
BatchNormalization = tf.keras.layers.BatchNormalization
Dropout = tf.keras.layers.Dropout
DenseVariational = tfp.layers.DenseVariational
MixtureNormal = tfp.layers.MixtureNormal

# ML configs #
units = 196 # Units in the network
n_layers = 3 # Number of hidden layers in the network
epochs = 700 # Epochs to train
batch_size = 1024 # Batch size

# MDN configs #
num_components = 7 # Number of components in the mixture
event_shape = [1] # Shape of the target
lr = 0.001
clipvalue = 0.5
clipnorm = 0.5
activation = tf.keras.layers.LeakyReLU()
opt = tf.keras.optimizers.Adam(lr=lr, clipvalue=clipvalue, clipnorm=clipnorm)

# Output config #
# Required input size for the mixture layer (num_components x event_shape x 3)
params_size = tfp.layers.MixtureNormal.params_size(num_components, event_shape)


def negloglik(y, p_y):
    return -p_y.log_prob(y)


def prior(kernel_size, bias_size=0, dtype=None):
    n = kernel_size + bias_size
    return tf.keras.Sequential([
        tfp.layers.VariableLayer(n, dtype=dtype),
        tfp.layers.DistributionLambda(
            lambda t: tfd.Independent(
                tfd.Normal(loc=t, scale=0.001), reinterpreted_batch_ndims=1
                )
            )
        ])


def posterior(kernel_size, bias_size, dtype=None):
    n = kernel_size + bias_size
    return tf.keras.Sequential([
        tfp.layers.VariableLayer(2 * n, dtype=dtype),
        tfp.layers.DistributionLambda(
            lambda t: tfd.Independent(
                tfd.Normal(loc=t[..., :n], scale=0.00001 + tf.nn.softplus(t[..., n:])), reinterpreted_batch_ndims=1
                )
            )
        ])


def Dense_Variational(train_sample_size=None, n_features=None):
    tf.keras.backend.clear_session()
    
    Input_Layer = Input(shape=(int(n_features),))
    l = DenseVariational(units, make_posterior_fn=posterior, make_prior_fn=prior,
                         kl_weight=1/train_sample_size, kl_use_exact=True, activation=activation
                         )(Input_Layer)
    l = BatchNormalization()(l)

    for _ in range(n_layers-1):
        l = DenseVariational(units, make_posterior_fn=posterior, make_prior_fn=prior,
                             kl_weight=1/train_sample_size, kl_use_exact=True, activation=activation
                             )(l)
        l = BatchNormalization()(l)

    l = Dense(params_size, activation=activation)(l)
    out = MixtureNormal(num_components, event_shape)(l)

    model = tf.keras.models.Model(inputs=Input_Layer, outputs=out)
    model.compile(optimizer=opt, loss=negloglik)

    return model
