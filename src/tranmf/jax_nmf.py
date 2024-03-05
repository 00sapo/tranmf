import jax
import jax.numpy as jnp
import optax

key = jax.random.PRNGKey(0)
h = jax.random.normal(key, (1000, 600))
w = jax.random.normal(key, (50, 1000))
v = jax.random.normal(key, (50, 600))


start_learning_rate = 1e-1
optimizer = optax.adam(start_learning_rate)
opt_state = optimizer.init((w, h))


@jax.jit
@jax.grad
def nmf_loss(matrices, v):
    w, h = matrices
    wh = jnp.dot(w, h)
    return optax.l2_loss(wh, v).sum()


for i in range(300):
    grads = nmf_loss((w, h), v)
    updates, opt_state = optimizer.update(grads, opt_state)
    w, h = optax.apply_updates((w, h), updates)

final_loss = optax.l2_loss(jnp.dot(w, h), v).sum()
print(final_loss)
