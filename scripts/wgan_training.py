# --------------------
# 2. WGAN Architecture
# --------------------

# WGAN loss function
def wasserstein_loss(y_true, y_pred):
    return K.mean(y_true * y_pred)

# Generator model
def build_generator(latent_dim, eeg_signal_length, num_classes):
    label = Input(shape=(1,), dtype='int32')
    label_embedding = Flatten()(Embedding(num_classes, latent_dim)(label))
    
    noise = Input(shape=(latent_dim,))
    model_input = multiply([noise, label_embedding])
    
    model = Sequential()
    model.add(Dense(256, input_dim=latent_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(1024))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(eeg_signal_length, activation='tanh'))  # Output is an EEG signal
    model.add(Reshape((eeg_signal_length,)))
    
    generator = Model([noise, label], model(model_input))
    return generator

# Critic (Discriminator) model
def build_critic(eeg_signal_length, num_classes):
    signal = Input(shape=(eeg_signal_length,))
    label = Input(shape=(1,), dtype='int32')
    
    label_embedding = Flatten()(Embedding(num_classes, eeg_signal_length)(label))
    model_input = multiply([signal, label_embedding])
    
    model = Sequential()
    model.add(Dense(1024, input_dim=eeg_signal_length))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(256))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(1))  # Output is a score
    
    critic = Model([signal, label], model(model_input))
    return critic

# Compile models
latent_dim = 100  # Latent space dimension
num_classes = 2  # Two classes: 0 (True) and 1 (False)
generator = build_generator(latent_dim, eeg_signal_length, num_classes)
critic = build_critic(eeg_signal_length, num_classes)

# WGAN optimizer
optimizer = RMSprop(lr=0.00005)

critic.compile(loss=wasserstein_loss, optimizer=optimizer)
generator.compile(loss=wasserstein_loss, optimizer=optimizer)
# --------------------
# 3. Training the WGAN
# --------------------

# Hyperparameters
epochs = 300
batch_size = 64
n_critic = 5  # Number of critic updates per generator update
clip_value = 0.01  # Clipping parameter for WGAN

# Create random noise for generator training
def generate_latent_points(latent_dim, n_samples):
    return np.random.randn(n_samples, latent_dim)

# Train WGAN
def train_wgan(generator, critic, eeg_data, labels, epochs, batch_size, latent_dim, n_critic, clip_value):
    d_losses, g_losses = [], []

    real = -np.ones((batch_size, 1))
    fake = np.ones((batch_size, 1))

    for epoch in range(epochs):
        for _ in range(n_critic):
            idx = np.random.randint(0, eeg_data.shape[0], batch_size)
            real_eeg = eeg_data[idx]
            real_labels = labels[idx]

            noise = generate_latent_points(latent_dim, batch_size)
            generated_eeg = generator.predict([noise, real_labels])

            d_loss_real = critic.train_on_batch([real_eeg, real_labels], real)
            d_loss_fake = critic.train_on_batch([generated_eeg, real_labels], fake)

            for layer in critic.layers:
                weights = layer.get_weights()
                weights = [np.clip(w, -clip_value, clip_value) for w in weights]
                layer.set_weights(weights)

        noise = generate_latent_points(latent_dim, batch_size)
        g_loss = generator.train_on_batch([noise, real_labels], real)

        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        d_losses.append(d_loss)
        g_losses.append(g_loss)

        if epoch % 100 == 0:
            print(f"Epoch {epoch}/{epochs} | D Loss: {d_loss:.4f} | G Loss: {g_loss:.4f}")
    
    return d_losses, g_losses

# Start training
d_losses, g_losses = train_wgan(generator, critic, eeg_data, labels, epochs, batch_size, latent_dim, n_critic, clip_value)

# Plot the loss curves
plt.plot(d_losses, label="Discriminator Loss")
plt.plot(g_losses, label="Generator Loss")
plt.legend()
plt.show()
