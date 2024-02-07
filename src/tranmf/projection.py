import mlflow
import numpy as np
from scipy.optimize import differential_evolution


def objective(x, projection_0, projection_1, projection_2, target=None):
    """
    Objective function for the projection optimization problem.

    It computes a loss given a candidate 3D matrix (`x`) and 3 2D projections
    (one for each dimension: `projection_0`, `projection_1`, `projection_2`).

    The candidate 3D matrix should satisfy the following constraints:
        - `x.sum(axis=1) = projection_0`
        - `x.sum(axis=2) = projection_1`
        - `x.sum(axis=3) = projection_2`
    As such, the loss is computed as the sum of the absolute differences between
    the x's projections and the target projections for each sample in the batch.

    Parameters:
        x : np.ndarray
            2D array of shape (len_candidate, num_candidates).
        projection_0 : np.ndarray, optional
            2D projection on axis 0.
        projection_1 : np.ndarray, optional
            2D projection on axis 1.
        projection_2 : np.ndarray, optional
            2D projection on axis 2.
        target : np.ndarray, optional
            Target 3D matrix to compare with. If provided, the loss is computed
            as the sum of the absolute differences between the candidate and the
            target.

    Returns:
        np.ndarray
            1D array of losses of shape (len_candidate, )
    """
    # Ensure x is 2D
    if x.ndim == 1:
        x = x[..., np.newaxis]

    # Check if each sample in x is a binary matrix
    # if not np.all(np.logical_or(x == 0, x == 1)):
    #     raise ValueError("Each sample in x should be a binary matrix")

    # Reshape x to have the third dimension
    shape = (
        projection_1.shape[0],
        projection_0.shape[0],
        projection_1.shape[1],
        x.shape[1],
    )
    x = x.reshape(shape)

    if target is not None:
        target = target[..., np.newaxis]
        x_ = x.copy()
        x_[x_ < 0.5] = 0
        x_[x_ >= 0.5] = 1
        real_loss = np.abs(target - x_).sum(axis=(0, 1, 2))
        if x.shape[-1] == 1:
            real_loss[0]

    if projection_0 is not None:
        projection_0 = projection_0[..., np.newaxis]
    if projection_1 is not None:
        projection_1 = projection_1[..., np.newaxis]
    if projection_2 is not None:
        projection_2 = projection_2[..., np.newaxis]

    # Compute projections for each sample
    x_proj_0 = x.sum(axis=0)
    x_proj_1 = x.sum(axis=1)
    x_proj_2 = x.sum(axis=2)
    loss_0 = loss_1 = loss_2 = np.zeros(x.shape[3])
    if projection_0 is not None:
        loss_0 = np.abs(projection_0 - x_proj_0).sum(axis=(0, 1))
    if projection_1 is not None:
        loss_1 = np.abs(projection_1 - x_proj_1).sum(axis=(0, 1))
    if projection_2 is not None:
        loss_2 = np.abs(projection_2 - x_proj_2).sum(axis=(0, 1))
    loss = loss_0 + loss_1 + loss_2

    mlflow.log_metric("loss_0_mean", np.mean(loss_0))
    mlflow.log_metric("loss_0_std", np.std(loss_0))
    mlflow.log_metric("loss_0_max", np.max(loss_0))
    mlflow.log_metric("loss_0_min", np.min(loss_0))
    mlflow.log_metric("loss_1_mean", np.mean(loss_1))
    mlflow.log_metric("loss_1_std", np.std(loss_1))
    mlflow.log_metric("loss_1_max", np.max(loss_1))
    mlflow.log_metric("loss_1_min", np.min(loss_1))
    mlflow.log_metric("loss_2_mean", np.mean(loss_2))
    mlflow.log_metric("loss_2_std", np.std(loss_2))
    mlflow.log_metric("loss_2_max", np.max(loss_2))
    mlflow.log_metric("loss_2_min", np.min(loss_2))

    mlflow.log_metric("loss_mean", np.mean(loss))
    mlflow.log_metric("loss_std", np.std(loss))
    mlflow.log_metric("loss_max", np.max(loss))
    mlflow.log_metric("loss_min", np.min(loss))
    mlflow.log_metric("real_loss_mean", np.mean(real_loss))
    mlflow.log_metric("real_loss_std", np.std(real_loss))
    mlflow.log_metric("real_loss_max", np.max(real_loss))
    mlflow.log_metric("real_loss_min", np.min(real_loss))

    if x.shape[-1] == 1:
        return loss[0]
    return loss


def find_3d_projection(projection_0, projection_1, projection_2, target):
    """
    Find a 3D matrix that satisfies the given projections.
    The problem is formulated as a differential evolution optimization problem
    where the objective function is the `objective` function defined above.

    Parameters
    ----------
    projection_0 : np.ndarray
        The first projection (2D matrix).
    projection_1 : np.ndarray
        The second projection (2D matrix).
    projection_2 : np.ndarray
        The third projection (2D matrix).

    Returns
    -------
    np.ndarray
        A 3D matrix that satisfies the given projections.
    np.float
        The loss of the estimated 3D matrix.
    """
    shape = projection_1.shape[0], projection_0.shape[0], projection_1.shape[1]
    S = shape[0] * shape[1] * shape[2]
    bounds = [(0, 1)] * S
    result = differential_evolution(
        objective,
        bounds,
        args=(projection_0, projection_1, projection_2, target),
        popsize=2,
        recombination=0.1,
        mutation=0.1,  # (0.0, 0.1),
        integrality=[True] * S,
        init=create_initial_population(shape, 20),
        vectorized=True,
        polish=False,
        disp=True,
        strategy=strategy,
    )
    return result


def _randtobest1(samples, mutation, population):
    """randtobest1bin, randtobest1exp"""
    r0, r1, r2 = samples[:3]
    bprime = np.copy(population[r0])
    return bprime


def strategy(candidate: int, population: np.ndarray, rng=None):
    """
    For scipy docs:

    > ... candidate is an integer specifying which entry of the population is being
    > evolved, population is an array of shape (S, N) containing all the population
    > members (where S is the total population size), and rng is the random number
    > generator being used within the solver. candidate will be in the range [0, S).
    > strategy must return a trial vector with shape (N,). The fitness of this trial
    > vector is compared against the fitness of population[candidate].

    This strategy generates a new trial which always sums to 1 along axis 2 and is made only of 0s and 1s.
    It first generate a sample as in best1bin, then it decides which elements force to 0.
    """
    # TODO: these override the other mutation and recombination parameters
    mutation = 0.2
    recombination = 0.2
    parameter_count = population.shape[1]

    # best1bin
    r0, r1 = rng.choice(population.shape[0], 2, replace=False)
    bprime = population[candidate] + mutation * (population[r0] - population[r1])

    # code for binomial crossover
    trial = np.copy(population[candidate])
    fill_point = rng.choice(parameter_count)
    crossovers = rng.uniform(size=parameter_count)
    crossovers = crossovers < recombination
    crossovers[fill_point] = True
    trial = np.where(crossovers, bprime, trial)

    # force the sum to 1 along axis 2
    # TODO: we need to accept this as a parameter... for now, compute the cubic root
    S = round(np.power(parameter_count, 1 / 3))
    trial = trial.reshape((S, S, S))
    for i in range(S):
        for j in range(S):
            non_zero = trial[i, j].nonzero()
            if non_zero[0].size > 0:
                idx = np.random.choice(non_zero[0])
                trial[i, j] = 0
                trial[i, j, idx] = 1
    return trial.reshape(-1)


def create_initial_population(shape, popsize):
    initial_population = []
    for p in range(popsize):
        initial_population.append(generate_3D_matrix(shape, np.random.rand()))

    initial_population = np.array(initial_population)
    initial_population = initial_population.reshape(popsize, -1)
    return initial_population


def generate_3D_matrix(shape, p):
    x = np.zeros(shape)
    # the sum along axis 2 must always be 1
    for i in range(S):
        for j in range(S):
            # with a probability of 0.5, we set the value to 1
            if np.random.rand() > p:
                # chose a random index to be 1 and the rest 0
                idx = np.random.randint(0, S)
                x[i, j, idx] = 1
    return x


if __name__ == "__main__":
    # Example usage
    import sys

    S = int(sys.argv[1])
    x = generate_3D_matrix((S, S, S), float(sys.argv[2]))
    print("True solution:\n", x)
    print("True solution shape:", x.shape)
    projection_0 = x.sum(axis=0)
    projection_1 = x.sum(axis=1)
    projection_2 = x.sum(axis=2)
    with mlflow.start_run():
        result = find_3d_projection(projection_0, projection_1, projection_2, x)
    proj3d = result.x.reshape(S, S, S)
    loss = result.fun
    print("----")
    print("Estimated solution:\n", proj3d)
    print("Estimated Loss:", loss)
    print("Real Loss:", np.abs(x - proj3d).sum())
    __import__("ipdb").set_trace()
