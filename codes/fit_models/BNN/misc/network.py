import tensorflow as tf
import tensorflow_probability as tfp; tfd = tfp.distributions
Input = tf.keras.layers.Input
Dense = tf.keras.layers.Dense
BatchNormalization = tf.keras.layers.BatchNormalization
Dropout = tf.keras.layers.Dropout
DenseVariational = tfp.layers.DenseVariational
MixtureNormal = tfp.layers.MixtureNormal

# ML configs #
Units = 196 # Units in the network
Num_Layers = 3 # Number of hidden layers in the network
Epochs = 600 # Epochs to train
Batch_Size = 1024 # Batch size

# MDN configs #
num_components = 7 # Number of components in the mixture
event_shape = [1] # Shape of the target
lr = 0.001
clipvalue = 0.5
clipnorm = 0.5
Activation = tf.keras.layers.LeakyReLU()
Opt = tf.keras.optimizers.Adam(lr=lr, clipvalue=clipvalue, clipnorm=clipnorm)

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


def Dense_Variational(Train_Sample_Size=None, Number_of_Features=None):
    tf.keras.backend.clear_session()
    
    Input_Layer = Input(shape=(int(Number_of_Features),))

    for i in range(Num_Layers):
        
        if i == 0:
            l = DenseVariational(
                Units, make_posterior_fn=posterior, make_prior_fn=prior,
                kl_weight=1/Train_Sample_Size, kl_use_exact=True, activation=Activation
                )(Input_Layer)
            l = BatchNormalization()(l)
            
        else:
            l = DenseVariational(
                Units, make_posterior_fn=posterior, make_prior_fn=prior,
                kl_weight=1/Train_Sample_Size, kl_use_exact=True, activation=Activation
                )(l)
            l = BatchNormalization()(l)

    l = Dense(params_size, activation=Activation)(l)

    out = MixtureNormal(num_components, event_shape)(l)

    model = tf.keras.models.Model(inputs=Input_Layer, outputs=out)
    model.compile(optimizer=Opt, loss=negloglik)

    return model
