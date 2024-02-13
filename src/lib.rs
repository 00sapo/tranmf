// import rand crate
use genevo::{genetic::Parents, operator::prelude::*, operator::CrossoverOp, prelude::*};
use memory_stats::memory_stats;
use ndarray::prelude::*;
use num::{abs, FromPrimitive, Integer, Signed, ToPrimitive};
use std::fmt::Debug;
use std::ops::Sub;

pub fn mem_stats() {
    if let Some(usage) = memory_stats() {
        println!(
            "Current memory usage (phys, virt), [MB]: {}, {}",
            usage.physical_mem / 2_usize.pow(20),
            usage.virtual_mem / 2_usize.pow(20)
        );
    } else {
        println!("Couldn't get the current memory usage :(");
    }
}

#[derive(Debug)]
struct Parameter {
    population_size: usize,
    generation_limit: u64,
    num_individuals_per_parents: usize,
    selection_ratio: f64,
    // num_crossover_points: usize,
    mutation_rate: f64,
    reinsertion_ratio: f64,
}

impl Default for Parameter {
    fn default() -> Self {
        Self {
            population_size: 24,
            generation_limit: 50,
            num_individuals_per_parents: 10,
            selection_ratio: 0.5,
            // num_crossover_points: 2,
            mutation_rate: 0.02, //25,
            reinsertion_ratio: 0.5,
        }
    }
}

trait MyNum:
    Integer + FromPrimitive + ToPrimitive + Signed + Debug + Clone + Send + Sync + PartialEq
{
}

impl MyNum for i8 {}

impl MyNum for i16 {}

impl MyNum for i32 {}

impl MyNum for i64 {}

impl MyNum for i128 {}

fn reconstruct_3d_array<T>(arr: &Array2<T>, shape2: usize) -> Array3<i8>
where
    T: MyNum,
{
    // construct the 3d array
    let mut arr3d = Array3::<i8>::default((arr.shape()[0], arr.shape()[1], shape2));
    for i in 0..arr.shape()[0] {
        for j in 0..arr.shape()[1] {
            let k = arr[[i, j]].to_usize().unwrap();
            if k < shape2 {
                arr3d[[i, j, k]] = 1;
            }
        }
    }
    arr3d
}

/// The genotype
#[derive(PartialEq, Debug, Clone)]
struct Genome<T>
where
    T: MyNum,
{
    arr: Array2<T>,
    projections: (Array2<T>, Array2<T>, Array2<T>),
}

impl<T> Genome<T>
where
    T: MyNum,
{
    fn from_arr(arr: Array2<T>, shape2: usize) -> Self {
        let arr3d = reconstruct_3d_array(&arr, shape2);
        let proj_0 = arr3d.sum_axis(Axis(0)).mapv(|x| T::from_i8(x).unwrap());
        let proj_1 = arr3d.sum_axis(Axis(1)).mapv(|x| T::from_i8(x).unwrap());
        let proj_2 = arr3d.sum_axis(Axis(2)).mapv(|x| T::from_i8(x).unwrap());
        Genome {
            arr,
            projections: (proj_0, proj_1, proj_2),
        }
    }

    /// Generate a 3D matrix of a given shape and fill it with 1s with a given probability.
    /// Then computes the arg-where of the ones and stores it into a genome.
    fn from_shape(
        shape: (usize, usize, usize),
        shape2_mask: &Array2<T>,
        rng: &mut impl Rng,
    ) -> Self {
        let mut arr3d = Array3::<bool>::default(shape);
        for i in 0..shape.0 {
            for j in 0..shape.1 {
                if shape2_mask[[i, j]] != T::zero() {
                    let k = rng.gen_range(0..shape.2);
                    arr3d[[i, j, k]] = true;
                }
            }
        }

        // convert the index of the 1s along the 2nd axis to their indices
        let arr = Array2::<T>::from_shape_fn((shape.0, shape.1), |(i, j)| {
            T::from_usize(
                (0..shape.2)
                    .position(|k| arr3d[[i, j, k]])
                    .unwrap_or(shape.2),
            )
            .unwrap()
        });

        // compute the projections
        let proj_0 = arr3d
            .mapv(|x| if x { T::one() } else { T::zero() })
            .sum_axis(Axis(0));
        let proj_1 = arr3d
            .mapv(|x| if x { T::one() } else { T::zero() })
            .sum_axis(Axis(1));
        let proj_2 = arr3d
            .mapv(|x| if x { T::one() } else { T::zero() })
            .sum_axis(Axis(2));

        Genome {
            arr,
            projections: (proj_0, proj_1, proj_2),
        }
    }

    pub fn project(&self, axis: usize) -> &Array2<T> {
        // convert the bool array to an array of 1 and 0 of type i32
        match axis {
            0 => &self.projections.0,
            1 => &self.projections.1,
            2 => &self.projections.2,
            _ => panic!("Invalid axis"),
        }
    }
}

impl<T> Genotype for Genome<T>
where
    T: MyNum,
{
    type Dna = T;
}

/// The fitness function for `Genome`s.
#[derive(Clone, Debug)]
struct FitnessCalc<T>
where
    T: MyNum,
{
    target_projections: (Array2<T>, Array2<T>, Array2<T>),
    max_diff: T,
}

impl<T> FitnessCalc<T>
where
    T: MyNum,
{
    pub fn new(target_projections: (Array2<T>, Array2<T>, Array2<T>)) -> Self {
        let genome_shape = (
            target_projections.1.shape()[0],
            target_projections.0.shape()[0],
            target_projections.0.shape()[1],
        );
        let max_diff = T::from_usize(genome_shape.0 * genome_shape.1 * genome_shape.2 * 2).unwrap();
        FitnessCalc {
            target_projections,
            max_diff,
        }
    }
}

impl<T> FitnessFunction<Genome<T>, usize> for FitnessCalc<T>
where
    T: MyNum,
{
    fn fitness_of(&self, genome: &Genome<T>) -> usize {
        // compute the projections of the genome
        let proj_0 = genome.project(0);
        let proj_1 = genome.project(1);
        // let proj_2 = genome.project(2);

        // take the sum of the absolute differences
        // between the projections and the target projections
        let diff_0 = (proj_0 - &self.target_projections.0).mapv(|x| abs(x)).sum();
        let diff_1 = (proj_1 - &self.target_projections.1).mapv(|x| abs(x)).sum();
        // let diff_2 = (proj_2 - &self.target_projections.2).mapv(|x| abs(x)).sum();
        let diff = diff_0 + diff_1; // + diff_2;

        // subtract from the maximum
        (self.max_diff.clone() - diff).to_usize().unwrap()
    }

    fn average(&self, fitness_values: &[usize]) -> usize {
        fitness_values.iter().sum::<usize>() / fitness_values.len()
    }

    fn highest_possible_fitness(&self) -> usize {
        self.max_diff.to_usize().unwrap()
    }

    fn lowest_possible_fitness(&self) -> usize {
        0
    }
}

impl<T> CrossoverOp<Genome<T>> for UniformCrossBreeder
where
    T: MyNum,
{
    fn crossover<R>(&self, parents: Parents<Genome<T>>, rng: &mut R) -> Vec<Genome<T>>
    where
        R: Rng + Sized,
    {
        let genome_shape = parents[0].arr.shape();
        let genome_shape_2 = parents[0].projections.0.shape()[1];
        let num_parents = parents.len();
        // breed one child for each partner in parents
        let mut children: Vec<Genome<T>> = Vec::with_capacity(num_parents);
        while num_parents > children.len() {
            let mut genome_arr = parents[0].arr.clone();
            for sp in 0..num_parents {
                // pick % of the indices to copy from parents[1]
                let num_indices = genome_shape[0] * genome_shape[1] / parents.len();
                for _ in 0..num_indices {
                    let i = rng.gen_range(0..genome_shape[0]);
                    let j = rng.gen_range(0..genome_shape[1]);
                    // assign the average value of the two parents
                    // // TODO: differential evolution here
                    // // respecting shape2 projection
                    // let avg: T = (parents[0].arr[[i, j]].clone() + parents[sp].arr[[i, j]].clone())
                    //     / T::from_usize(2).unwrap();
                    // genome_arr[[i, j]] = avg;
                    // copy the value from the 2nd parent
                    genome_arr
                        .slice_mut(s![i, j])
                        .assign(&parents[sp].arr.slice(s![i, j]));
                }
            }
            children.push(Genome::from_arr(genome_arr, genome_shape_2));
        }
        children
    }
}

impl<T> RandomGenomeMutation for Genome<T>
where
    T: MyNum,
{
    type Dna = T;

    fn mutate_genome<R>(
        genome: Self,
        mutation_rate: f64,
        min_value: &<Self as Genotype>::Dna,
        max_value: &<Self as Genotype>::Dna,
        rng: &mut R,
    ) -> Self
    where
        R: Rng + Sized,
    {
        // randomly mutate the genome
        let min = (*min_value).to_usize().unwrap();
        let max = (*max_value).to_usize().unwrap();

        let mut new_genome = genome.clone();
        for i in 0..genome.arr.shape()[0] {
            for j in 0..genome.arr.shape()[1] {
                if rng.gen_bool(mutation_rate) && (genome.projections.2[[i, j]] != T::zero()) {
                    new_genome.arr[[i, j]] = T::from_usize(rng.gen_range(min..max)).unwrap();
                }
            }
        }
        new_genome
    }
}

impl<T> Sub for Genome<T>
where
    T: MyNum,
{
    type Output = Array3<f32>;
    fn sub(self, other: Self) -> Self::Output {
        // reconstruct the 3d array
        let arr3d_1 = reconstruct_3d_array(&self.arr, self.projections.2.shape()[1]);
        let arr3d_2 = reconstruct_3d_array(&other.arr, other.projections.2.shape()[1]);
        (arr3d_1 - arr3d_2).mapv(|x| abs(x) as f32)
    }
}

fn new_population<T>(
    size: usize,
    shape: (usize, usize, usize),
    shape2_mask: &Array2<T>,
    rng: &mut impl Rng,
) -> Population<Genome<T>>
where
    T: MyNum,
{
    // generate a vec of genomes with p distributed from 0 to 1 with step 1 / size
    let mut pop = Vec::with_capacity(size);
    for _ in 0..size {
        pop.push(Genome::from_shape(shape, shape2_mask, rng));
    }
    Population::with_individuals(pop)
}

// Exported function for Python
#[no_mangle]
pub extern "C" fn run_genetic_algorithm() {
    let shape = (1000, 1000, 100);
    let rng = &mut rand::thread_rng();
    let p = 0.5;
    // generate a random boolean matrix as a mask for the 3rd axis
    let shape2_mask =
        Array2::<i32>::from_shape_fn(
            (shape.0, shape.1),
            |(_, _)| {
                if rng.gen_bool(p) {
                    1
                } else {
                    0
                }
            },
        );

    let ground_truth = Genome::<i32>::from_shape(shape, &shape2_mask, rng);
    let fitness_calc = FitnessCalc::new(ground_truth.projections.clone());

    let params = Parameter::default();

    let initial_population = new_population(
        params.population_size,
        shape,
        &ground_truth.projections.2,
        rng,
    );

    let mut projection_sim = simulate(
        genetic_algorithm()
            .with_evaluation(fitness_calc.clone())
            .with_selection(MaximizeSelector::new(
                params.selection_ratio,
                params.num_individuals_per_parents,
            ))
            .with_crossover(UniformCrossBreeder::default())
            .with_mutation(RandomValueMutator::new(
                params.mutation_rate,
                0i32,
                shape.2 as i32,
            ))
            .with_reinsertion(ElitistReinserter::new(
                fitness_calc.clone(),
                true,
                params.reinsertion_ratio,
            ))
            .with_initial_population(initial_population)
            .build(),
    )
    .until(or(
        FitnessLimit::new(fitness_calc.highest_possible_fitness()),
        GenerationLimit::new(params.generation_limit),
    ))
    .build();
    mem_stats();

    println!("Starting projection optimization with: {:?}", params);

    loop {
        let result = projection_sim.step();
        match result {
            Ok(SimResult::Intermediate(step)) => {
                let evaluated_population = step.result.evaluated_population;
                let best_solution = step.result.best_solution;
                let best_proj2 = best_solution.solution.genome.project(2);
                let diff = (ground_truth.clone() - best_solution.solution.genome.clone()).sum()
                    as f32
                    / (shape.0 * shape.1 * shape.2) as f32;
                let diff_proj_2 = (best_proj2 - &fitness_calc.target_projections.2)
                    .mapv(|x| x.abs())
                    .sum() as f32
                    / ((shape.0 * shape.1) as f32);
                println!(
                    "Step: generation: {}, average_fitness: {}, \
                    best fitness: {}, real diff: {}, diff proj 2: {}",
                    step.iteration,
                    evaluated_population.average_fitness(),
                    best_solution.solution.fitness as f32 / fitness_calc.max_diff as f32,
                    diff,
                    diff_proj_2
                );
                mem_stats();
            }
            Ok(SimResult::Final(step, _processing_time, duration, stop_reason)) => {
                let best_solution = step.result.best_solution;
                let diff = (ground_truth.clone() - best_solution.solution.genome.clone()).sum();
                println!("{}", stop_reason);
                println!(
                    "Final result after {}: generation: {}, \
                    best solution with fitness {} found in generation {}, real diff: {}",
                    duration,
                    step.iteration,
                    best_solution.solution.fitness,
                    best_solution.generation,
                    diff
                );
                // println!("Final result: {:?}", best_solution.solution.genome.arr);
                // println!("Ground truth: {:?}", ground_truth.arr);
                break;
            }
            Err(error) => {
                println!("{}", error);
                break;
            }
        }
    }
}
