// import rand crate
use genevo::{
    operator::prelude::*,  prelude::*, types::fmt::Display, operator::CrossoverOp, genetic::Parents, operator::MutationOp, operator::GeneticOperator
};
use memory_stats::memory_stats;
use ndarray::prelude::*;
// impor ttrait Sub
use std::ops::Sub;

pub fn mem_stats() {
    if let Some(usage) = memory_stats() {
        println!("Current memory usage (phys, virt), [MB]: {}, {}", usage.physical_mem / 2_usize.pow(20), usage.virtual_mem / 2_usize.pow(20));
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
    mutation_rate: (f64, f64),
    reinsertion_ratio: f64,
}

impl Default for Parameter {
    fn default() -> Self {
        Self {
            population_size: 12,
            generation_limit: 1000,
            num_individuals_per_parents: 2,
            selection_ratio: 0.8,
            // num_crossover_points: 2,
            mutation_rate: (0.0, 1e-4),
            reinsertion_ratio: 0.5,
        }
    }
}


/// The genotype
#[derive(PartialEq, Clone, Debug)]
struct Genome {
    arr: Array3<bool>
}

impl Genome{
    // pub fn new(shape: &[usize]) -> Self {
    //     Genome{arr: Array3::<bool>::default((shape[0], shape[1], shape[2]))}
    // }

    pub fn project(&self, axis: usize) -> Array2<i32> {
        // convert the bool array to an array of 1 and 0 of type i32
        let arr = self.arr.mapv(|x| if x {1i32} else {0i32});
        arr.sum_axis(Axis(axis))
    }
}


impl Genotype for Genome {
    type Dna = bool;
}



/// The fitness function for `Genome`s.
#[derive(Clone, Debug)]
struct FitnessCalc {
    target_projections: (Array2<i32>, Array2<i32>, Array2<i32>),
    max_diff: usize,
    }

impl FitnessCalc {
    pub fn new(target_projections: (Array2<i32>, Array2<i32>, Array2<i32>)) -> Self {
        let genome_shape = (target_projections.1.shape()[0], target_projections.0.shape()[0], target_projections.0.shape()[1]);
        let max_diff = genome_shape.0 * genome_shape.1 * (2 * genome_shape.2 + 1);
        FitnessCalc { target_projections, max_diff }
    }
}

impl FitnessFunction<Genome, usize> for FitnessCalc
{
    fn fitness_of(&self, genome: &Genome) -> usize {
        // compute the projections of the genome
        let proj_0 = genome.project(0);
        let proj_1 = genome.project(1);
        let proj_2 = genome.project(2);

        // take the sum of the absolute differences
        // between the projections and the target projections
        let diff_0 = (proj_0 - &self.target_projections.0).mapv(|x| x.abs()).sum();
        let diff_1 = (proj_1 - &self.target_projections.1).mapv(|x| x.abs()).sum();
        let diff_2 = (proj_2 - &self.target_projections.2).mapv(|x| x.abs()).sum();
        let diff = (diff_0 + diff_1 + diff_2) as usize;

        // subtract from the maximum
        self.max_diff - diff
    }

    fn average(&self, fitness_values: &[usize]) -> usize {
        fitness_values.iter().sum::<usize>() / fitness_values.len()
    }

    fn highest_possible_fitness(&self) -> usize {
        self.max_diff
    }

    fn lowest_possible_fitness(&self) -> usize {
        0
    }
}

/// Generate a 3D matrix of a given shape and fill it with 1s with a given probability.
fn generate_genome(shape: (usize, usize, usize), p: f64) -> Genome {
    let mut x = Array3::<bool>::default(shape);
    for i in 0..shape.0 {
        for j in 0..shape.1 {
            if rand::random::<f64>() < p {
                let idx = rand::random::<usize>() % shape.2;
                x[[i, j, idx]] = true;
            }
        }
    }
    Genome{arr: x}
}

#[derive(Debug, Clone)]
struct RandomValueMutatorWithDithering {
    mutation_rate: (f64, f64),
}

impl GeneticOperator for RandomValueMutatorWithDithering {
    fn name() -> String {
        "RandomValueMutatorWithDithering".to_string()
    }
}

impl MutationOp<Genome> for RandomValueMutatorWithDithering {
    fn mutate<R>(&self, genome: Genome, rng: &mut R) -> Genome
    where
        R: Rng + Sized,
    {
        let mut mutated_genome = genome.clone();
        let shape = genome.arr.shape();
        // compute the indices of the genome to mutate
        // by chosing mutation_rate * genome_size indices
        // Perform dithering to avoid bias
        let mut_rate_dithering = rng.gen_range(self.mutation_rate.0..self.mutation_rate.1);
        let num_indices = (mut_rate_dithering * (shape[0] * shape[1] * shape[2]) as f64) as usize;
        for _ in 0..num_indices {
            let i = rng.gen_range(0..shape[0]);
            let j = rng.gen_range(0..shape[1]);
            
            // find any existing true element and flip it
            let k = mutated_genome.arr.slice(s![i, j, ..]).iter().position(|&x| x);

            if let Some(k) = k {
                mutated_genome.arr[[i, j, k]] = !mutated_genome.arr[[i, j, k]];
            } else {
                let k = rng.gen_range(0..shape[2]);
                mutated_genome.arr[[i, j, k]] = !mutated_genome.arr[[i, j, k]];
            }
            // mutated_genome.arr[[i, j, k]] = !mutated_genome.arr[[i, j, k]];
        }
        mutated_genome
    }
}


/// Custom genome builder
struct MyGenomeBuilder
    {
    shape: (usize, usize, usize),
    }

impl GenomeBuilder<Genome> for MyGenomeBuilder {
    fn build_genome<R>(&self, _: usize, rng: &mut R) -> Genome
        where
            R: Rng + Sized {
        generate_genome(self.shape, rng.gen_range(0.0..1.0))
    }
}

impl CrossoverOp<Genome> for UniformCrossBreeder {
        fn crossover<R>(&self, parents: Parents<Genome>, rng: &mut R) -> Vec<Genome>
        where
            R: Rng + Sized,
        {
            let genome_shape = parents[0].arr.shape();
            let num_parents = parents.len();
            // breed one child for each partner in parents
            let mut children: Vec<Genome> = Vec::with_capacity(num_parents);
            while num_parents > children.len() {
                let mut genome = parents[0].clone();
                for sp in 0..num_parents {
                    // pick % of the indices to copy from parents[1]
                    let num_indices = genome_shape[0] * genome_shape[1] / parents.len();
                    for _ in 0..num_indices {
                        let i = rng.gen_range(0..genome_shape[0]);
                        let j = rng.gen_range(0..genome_shape[1]);
                        // copy the value from the 2nd parent
                        genome.arr.slice_mut(s![i, j, ..]).assign(&parents[sp].arr.slice(s![i, j, ..]));
                    }
                }
                children.push(genome);
            }
            children
        }
    }

// implement subtraction for genomes
impl Sub for Genome {
    type Output = Array3<i32>;
    fn sub(self, other: Self) -> Self::Output {
        let diff: Array3<i32> = self.arr.mapv(|x| if x {1i32} else {0i32}) - other.arr.mapv(|x| if x {1i32} else {0i32});
        diff.mapv(|x| x.abs())
    }
}

// Exported function for Python
#[no_mangle]
pub extern "C" fn run_genetic_algorithm() {

    let shape = (200, 200, 200);
    let p = 0.8;
    let ground_truth: Genome = generate_genome(shape, p);
    let proj_0 = ground_truth.project(0);
    let proj_1 = ground_truth.project(1);
    let proj_2 = ground_truth.project(2);
    let fitness_calc = FitnessCalc::new((proj_0, proj_1, proj_2));

    let params = Parameter::default();

    let initial_population: Population<Genome> = build_population()
        .with_genome_builder(MyGenomeBuilder { shape })
        .of_size(params.population_size)
        .uniform_at_random();

    let mut projection_sim = simulate(
        genetic_algorithm()
            .with_evaluation(fitness_calc.clone())
            .with_selection(MaximizeSelector::new(
                params.selection_ratio,
                params.num_individuals_per_parents,
            ))
            .with_crossover(UniformCrossBreeder::default())
            .with_mutation(RandomValueMutatorWithDithering{mutation_rate: params.mutation_rate})
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
                let diff = (ground_truth.clone() - best_solution.solution.genome.clone()).sum() as f32 / (ground_truth.arr.len() as f32);
                let diff_proj_2 = (best_proj2 - &fitness_calc.target_projections.2).mapv(|x| x.abs()).sum() as f32 / ((shape.0 * shape.1) as f32);
                println!(
                    "Step: generation: {}, average_fitness: {}, \
                    best fitness: {}, duration: {}, processing_time: {}, real diff: {}, diff proj 2: {}",
                    step.iteration,
                    evaluated_population.average_fitness(),
                    best_solution.solution.fitness as f32 / fitness_calc.max_diff as f32,
                    step.duration.fmt(),
                    step.processing_time.fmt(),
                    diff,
                    diff_proj_2
                );
                mem_stats();
            },
            Ok(SimResult::Final(step, processing_time, duration, stop_reason)) => {
                let best_solution = step.result.best_solution;
                let diff = (ground_truth.clone() - best_solution.solution.genome.clone()).sum();
                println!("{}", stop_reason);
                println!(
                    "Final result after {}: generation: {}, \
                    best solution with fitness {} found in generation {}, processing_time: {}, real diff: {}",
                    duration.fmt(),
                    step.iteration,
                    best_solution.solution.fitness,
                    best_solution.generation,
                    processing_time.fmt(),
                    diff
                );
                // println!("Final result: {:?}", best_solution.solution.genome.arr);
                // println!("Ground truth: {:?}", ground_truth.arr);
                break;
                            
            },
            Err(error) => {
                println!("{}", error);
                break;
            },
        }
    }
}
