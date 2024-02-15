// import rand crate
use genevo::{
    genetic::Parents, operator::prelude::*, operator::CrossoverOp, operator::GeneticOperator,
    prelude::*,
};
use memory_stats::memory_stats;
use ndarray::prelude::*;
use std::fmt::Debug;
use std::io;

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

type Int = i32;

#[derive(Debug)]
struct Parameter {
    population_size: usize,
    generation_limit: u64,
    num_individuals_per_parents: usize,
    selection_ratio: f64,
    crossover_scale: f32,
    crossover_fitness_ratio: f32,
    crossover_fitness_prob: f32,
    recombination: (f32, f32),
    mutation_rate: f64,
    reinsertion_ratio: f64,
}

impl Default for Parameter {
    fn default() -> Self {
        Self {
            population_size: 5000,
            generation_limit: 50000,
            num_individuals_per_parents: 1000, // musst be 3 for DifferentialCrossover
            selection_ratio: 0.5,
            crossover_scale: 0.7,
            crossover_fitness_ratio: 1.00,
            crossover_fitness_prob: 0.5,
            recombination: (0.1, 0.3),
            mutation_rate: 0.001,
            reinsertion_ratio: 0.5,
        }
    }
}

fn reconstruct_3d_array(arr: &Array2<Int>, shape2: usize) -> Array3<Int> {
    // construct the 3d array
    Array3::<Int>::from_shape_fn((arr.shape()[0], arr.shape()[1], shape2), |(i, j, k)| {
        if arr[[i, j]] == k as Int {
            1
        } else {
            0
        }
    })
}

fn arr3d_to_map(arr: &Array3<Int>) -> Array2<Int> {
    let shape = arr.shape();
    // convert the index of the 1s along the 2nd axis to their indices
    Array2::<Int>::from_shape_fn((shape[0], shape[1]), |(i, j)| {
        for k in 0..shape[2] {
            if arr[[i, j, k]] != Int::zero() {
                return k as Int;
            }
        }
        shape[2] as Int
    })
}

/// The genotype
#[derive(PartialEq, Debug, Clone)]
struct Genome {
    arr: Array2<Int>,
    projections: (Array2<Int>, Array2<Int>, Array2<Int>),
}

impl Genome {
    fn from_arr(arr: Array2<Int>, shape2: usize) -> Self {
        let arr3d = reconstruct_3d_array(&arr, shape2);
        let proj_0 = arr3d.sum_axis(Axis(0));
        let proj_1 = arr3d.sum_axis(Axis(1));
        let proj_2 = arr3d.sum_axis(Axis(2));
        Genome {
            arr,
            projections: (proj_0, proj_1, proj_2),
        }
    }

    /// Generate a 3D matrix of a given shape and fill it with 1s with a given probability.
    /// Then computes the arg-where of the ones and stores it into a genome.
    fn from_shape(
        shape: (usize, usize, usize),
        shape2_mask: &Array2<Int>,
        rng: &mut impl Rng,
    ) -> Self {
        let mut arr3d = Array3::<Int>::zeros(shape);
        for i in 0..shape.0 {
            for j in 0..shape.1 {
                if shape2_mask[[i, j]] != Int::zero() {
                    let k = rng.gen_range(0..shape.2);
                    arr3d[[i, j, k]] = 1;
                }
            }
        }

        // compute the projections
        let proj_0 = arr3d.sum_axis(Axis(0));
        let proj_1 = arr3d.sum_axis(Axis(1));
        let proj_2 = arr3d.sum_axis(Axis(2));

        let arr = arr3d_to_map(&arr3d);

        Genome {
            arr,
            projections: (proj_0, proj_1, proj_2),
        }
    }

    pub fn project(&self, axis: usize) -> &Array2<Int> {
        // convert the bool array to an array of 1 and 0 of type i32
        match axis {
            0 => &self.projections.0,
            1 => &self.projections.1,
            2 => &self.projections.2,
            _ => panic!("Invalid axis"),
        }
    }
}

impl Genotype for Genome {
    type Dna = Int;
}

#[derive(Clone, Debug)]
struct TargetProjections {
    proj_0: Array2<Int>,
    proj_1: Array2<Int>,
    proj_2: Array2<Int>,
}

impl TargetProjections {
    fn new(ground_truth: &Genome) -> Self {
        let proj_0 = ground_truth.project(0).clone();
        let proj_1 = ground_truth.project(1).clone();
        let proj_2 = ground_truth.project(2).clone();
        TargetProjections {
            proj_0,
            proj_1,
            proj_2,
        }
    }
    fn fitness(&self, genome: &Genome) -> (Array2<Int>, Array2<Int>) {
        // compute the projections of the genome
        let proj_0 = genome.project(0);
        let proj_1 = genome.project(1);
        // let proj_2 = genome.project(2);
        // take the sum of the absolute differences
        // between the projections and the target projections
        let diff_0 = proj_0 - &self.proj_0;
        let diff_1 = proj_1 - &self.proj_1;
        // let diff_2 = (proj_2 - &target_projections.2).mapv(|x| abs(x)).sum();
        (diff_0, diff_1)
    }
}

/// The fitness function for `Genome`s.
#[derive(Clone, Debug)]
struct FitnessStruct {
    target_projections: TargetProjections,
    max_diff: Int,
}

impl FitnessStruct {
    pub fn new(target_projections: (Array2<Int>, Array2<Int>, Array2<Int>)) -> Self {
        let genome_shape = (
            target_projections.1.shape()[0],
            target_projections.0.shape()[0],
            target_projections.0.shape()[1],
        );
        let max_diff = (genome_shape.0 * genome_shape.1 * genome_shape.2 * 2) as Int;
        let target_projections = TargetProjections {
            proj_0: target_projections.0.clone(),
            proj_1: target_projections.1.clone(),
            proj_2: target_projections.2.clone(),
        };
        FitnessStruct {
            target_projections,
            max_diff,
        }
    }
}

impl FitnessFunction<Genome, usize> for FitnessStruct {
    fn fitness_of(&self, genome: &Genome) -> usize {
        let (diff0, diff1) = self.target_projections.fitness(genome);
        // subtract from the maximum
        let diff = diff0.mapv(|x| x.abs()).sum() + diff1.mapv(|x| x.abs()).sum();
        (self.max_diff - diff) as usize
    }

    fn average(&self, fitness_values: &[usize]) -> usize {
        fitness_values.iter().sum::<usize>() / fitness_values.len()
    }

    fn highest_possible_fitness(&self) -> usize {
        self.max_diff as usize
    }

    fn lowest_possible_fitness(&self) -> usize {
        0
    }
}

#[derive(Clone, Debug)]
struct DifferentialCrossover {
    recombination: (f32, f32),
    crossover_scale: f32,
    crossover_fitness_ratio: f32,
    crossover_fitness_prob: f32,
    target_projections: TargetProjections,
}

impl GeneticOperator for DifferentialCrossover {
    fn name() -> String {
        "DifferentialCrossover".to_string()
    }
}

impl CrossoverOp<Genome> for DifferentialCrossover {
    fn crossover<R>(&self, parents: Parents<Genome>, rng: &mut R) -> Vec<Genome>
    where
        R: Rng + Sized,
    {
        let genome_shape = parents[0].arr.shape();
        let genome_shape_2 = parents[0].projections.0.shape()[1];
        let num_parents = parents.len();
        // breed one child for each partner in parents
        let mut children: Vec<Genome> = Vec::with_capacity(num_parents);
        while num_parents > children.len() {
            let mut genome_arr = parents[0].arr.clone();
            let fitness = self.target_projections.fitness(&parents[0]);
            for sp in 0..num_parents {
                // pick % of the indices to copy from parents[1]
                let num_indices = genome_shape[0] * genome_shape[1] / parents.len();
                for _ in 0..num_indices {
                    let i = rng.gen_range(0..genome_shape[0]);
                    let j = rng.gen_range(0..genome_shape[1]);
                    // assign the average value of the two parents
                    // TODO: differential evolution here
                    // respecting shape2 projection
                    // let avg: Int = (parents[0].arr[[i, j]].clone() - parents[sp].arr[[i, j]].clone())
                    //     * T::from_f64(0.5).unwrap()
                    //     + parents[0].arr[[i, j]].clone();
                    // genome_arr[[i, j]] = avg;
                    // copy the value from the 2nd parent
                    genome_arr
                        .slice_mut(s![i, j])
                        .assign(&parents[sp].arr.slice(s![i, j]));
                }
            }

            // adjust_genome(
            //     &mut genome_arr,
            //     fitness,
            //     self.crossover_fitness_ratio,
            //     self.crossover_fitness_prob,
            //     rng,
            // );

            children.push(Genome::from_arr(genome_arr, genome_shape_2));
        }
        children
    }
}

fn adjust_genome(
    genome_arr: &mut Array2<Int>,
    fitness: (Array2<Int>, Array2<Int>),
    ratio: f32,
    prob: f32,
    rng: &mut impl Rng,
) {
    // sort the indices by the difference
    let indices = argsort(&fitness.0, &fitness.1);
    let end = (indices.len() as f32 * ratio).round() as usize;
    for idx in &indices[0..end] {
        if rng.gen_bool(prob as f64) {
            let diff = (fitness.0[[idx.0, idx.2]] + fitness.1[[idx.1, idx.2]]) as f64 / 2f64;
            genome_arr[[idx.1, idx.0]] += diff.round() as Int;
        }
    }
}

impl RandomGenomeMutation for Genome {
    type Dna = Int;

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

        let mut new_genome = genome.clone();
        for i in 0..genome.arr.shape()[0] {
            for j in 0..genome.arr.shape()[1] {
                if rng.gen_bool(mutation_rate) && (genome.projections.2[[i, j]] != Int::zero()) {
                    new_genome.arr[[i, j]] = rng.gen_range(*min_value..*max_value);
                }
            }
        }
        new_genome
    }
}

/// returns a vector of (usize, usize, usize) representing the indices
/// sorted of the two input arrays
/// input arrays must have the second dimension of the same size, and it will be in the third
/// position of the output
fn argsort(data1: &Array2<Int>, data2: &Array2<Int>) -> Vec<(usize, usize, usize)> {
    let shape1 = data1.shape();
    let shape2 = data2.shape();
    assert!(
        shape1[1] == shape2[1],
        "The second dimension must be the same"
    );

    let mut indices: Vec<(usize, usize, usize)> = (0..shape1[0])
        .flat_map(|i| (0..shape2[0]).flat_map(move |j| (0..shape1[1]).map(move |k| (i, j, k))))
        .collect();
    indices.sort_by_key(|&idx| data1[[idx.0, idx.2]] + data2[[idx.1, idx.2]]);
    indices
}

fn new_population(
    size: usize,
    projections: &(Array2<Int>, Array2<Int>, Array2<Int>),
    rng: &mut impl Rng,
) -> Vec<Genome> {
    let mut projections_0 = projections.0.clone();
    let mut projections_1 = projections.1.clone();
    // solve the problem with a greedy approach
    let shape = (
        projections_1.shape()[0],
        projections_0.shape()[0],
        projections_0.shape()[1],
    );
    let mut root = Array2::<Int>::from_elem((shape.0, shape.1), shape.2 as Int);
    // println!("Prj0\n{:?}", projections_0);
    // println!("Prj1\n{:?}", projections_1);
    'outer: loop {
        let mut indices = argsort(&projections_1, &projections_0);
        indices.reverse();
        // println!("Indices\n{:?}", indices);
        let mut breaking = false;
        'inner: for idx in indices {
            // println!("Idx{:?}", idx);
            if projections.2[[idx.0, idx.1]] == 1
                && projections_0[[idx.1, idx.2]] != 0
                && projections_1[[idx.0, idx.2]] != 0
            {
                root[[idx.0, idx.1]] = idx.2 as Int;
                projections_0[[idx.1, idx.2]] -= 1;
                projections_1[[idx.0, idx.2]] -= 1;
                breaking = true;
                break 'inner;
            }
        }
        // println!("Root\n{:?}", root);
        // println!("Prj0\n{:?}", projections_0);
        // println!("Prj1\n{:?}", projections_1);
        if !breaking {
            break 'outer;
        }
    }

    // // force values != shape.2 where there is 1 in the projections.2
    // projections.2.indexed_iter().for_each(|(idx, &x)| {
    //     if x == 1 && root[[idx.0, idx.1]] == shape.2 as Int {
    //         root[[idx.0, idx.1]] = shape.2 as Int / 2;
    //     }
    // });

    // generate a vec of genomes with p distributed from 0 to 1 with step 1 / size
    let mut pop = Vec::with_capacity(size);
    pop.push(Genome::from_arr(root.clone(), shape.2));
    for p in 1..size {
        let mut genome = root.clone();
        for i in 0..shape.0 {
            for j in 0..shape.1 {
                if rng.gen_bool(1f64 / size as f64 * p as f64)
                    && (projections.2[[i, j]] != Int::zero())
                {
                    genome[[i, j]] = rng.gen_range(0..shape.2 as Int);
                }
            }
        }
        pop.push(Genome::from_arr(genome, shape.2));
    }
    pop
}

// Exported function for Python
#[no_mangle]
pub extern "C" fn run_genetic_algorithm() {
    let shape = (10, 10, 10);
    let rng = &mut rand::thread_rng();
    let p = 0.5;
    // generate a random boolean
    let shape2_mask =
        Array2::<Int>::from_shape_fn(
            (shape.0, shape.1),
            |(_, _)| {
                if rng.gen_bool(p) {
                    1
                } else {
                    0
                }
            },
        );
    let ground_truth = Genome::from_shape(shape, &shape2_mask, rng);
    let target_projections = TargetProjections::new(&ground_truth);

    let fitness_obj = FitnessStruct::new(ground_truth.projections.clone());

    let params = Parameter::default();

    println!("Projections 2\n{:?}", ground_truth.projections.2);
    println!("Target\n{:?}", ground_truth.arr);
    let initial_population = new_population(params.population_size, &ground_truth.projections, rng);

    // print the first element of initial_population and its diff from ground_truth
    let first_genome = &initial_population[0];
    let diff = ground_truth
        .arr
        .iter()
        .zip(first_genome.arr.iter())
        .map(|(&x, &y)| if x != y { 1 } else { 0 })
        .sum::<i32>();
    println!("{:?}", ground_truth.projections.2);
    println!("{:?}", ground_truth.arr);
    println!("{:?}", first_genome.arr);
    println!("Initial diff: {}", diff);
    // return;

    //////////////////////////////////////////////////
    let mut projection_sim = simulate(
        genetic_algorithm()
            .with_evaluation(fitness_obj.clone())
            .with_selection(MaximizeSelector::new(
                params.selection_ratio,
                params.num_individuals_per_parents,
            ))
            .with_crossover(DifferentialCrossover {
                recombination: params.recombination,
                crossover_scale: params.crossover_scale,
                crossover_fitness_ratio: params.crossover_fitness_ratio,
                crossover_fitness_prob: params.crossover_fitness_prob,
                target_projections: target_projections.clone(),
            })
            .with_mutation(RandomValueMutator::new(
                params.mutation_rate,
                0 as Int,
                shape.2 as Int,
            ))
            .with_reinsertion(ElitistReinserter::new(
                fitness_obj.clone(),
                true,
                params.reinsertion_ratio,
            ))
            .with_initial_population(Population::with_individuals(initial_population))
            .build(),
    )
    .until(or(
        FitnessLimit::new(fitness_obj.highest_possible_fitness()),
        GenerationLimit::new(params.generation_limit),
    ))
    .build();

    /////////////////////////////////////////////////
    mem_stats();

    println!("Starting projection optimization with: {:?}", params);

    loop {
        let result = projection_sim.step();
        match result {
            Ok(SimResult::Intermediate(step)) => {
                let evaluated_population = step.result.evaluated_population;
                let best_solution = step.result.best_solution;
                let diff = ground_truth
                    .arr
                    .iter()
                    .zip(best_solution.solution.genome.arr.iter())
                    .map(|(&x, &y)| if x != y { 1f64 } else { 0f64 })
                    .sum::<f64>()
                    / (ground_truth.projections.2.len() as f64);
                println!(
                    "Step: generation: {}, average_fitness: {}, \
                    best fitness: {:?} real diff: {:?}",
                    step.iteration,
                    evaluated_population.average_fitness(),
                    best_solution.solution.fitness as f64 / fitness_obj.max_diff as f64,
                    diff,
                );
                mem_stats();
            }
            Ok(SimResult::Final(step, _processing_time, duration, stop_reason)) => {
                let best_solution = step.result.best_solution;
                let diff = ground_truth
                    .arr
                    .iter()
                    .zip(best_solution.solution.genome.arr.iter())
                    .map(|(&x, &y)| if x != y { 1 } else { 0 })
                    .sum::<i32>();
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
                // if diff != 0 {
                //     println!(
                //         "{:?}",
                //         reconstruct_3d_array(&best_solution.solution.genome.arr, shape.2)
                //     );
                //     println!("{:?}", reconstruct_3d_array(&ground_truth.arr, shape.2));
                //     println!("{:?}", best_solution.solution.genome.projections);
                //     println!("...");
                //     println!("{:?}", ground_truth.projections);
                //     println!("...");
                //     println!("{:?}", best_solution.solution.genome.arr);
                //     println!("...");
                //
                //     println!("{:?}", shape2_mask);
                //     println!("{:?}", ground_truth.arr);
                //     let new_map = reconstruct_3d_array(&ground_truth.arr, shape.2);
                //     println!("{:?}", new_map.sum_axis(Axis(2)));
                //     let new_ground_truth = Genome::from_arr(arr3d_to_map(&new_map), shape.2).arr;
                //     println!("{:?}", new_ground_truth);
                // }
                break;
            }
            Err(error) => {
                println!("{}", error);
                break;
            }
        }
    }
}
