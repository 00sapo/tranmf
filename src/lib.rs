// import rand crate
use genevo::{
    genetic::Parents, operator::prelude::*, operator::CrossoverOp, operator::GeneticOperator,
    prelude::*,
};
use memory_stats::memory_stats;
use ndarray::prelude::*;
use std::fmt::Debug;
use std::ops::{Index, Sub};

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
    crossover_scale: f32,
    recombination: f32,
    mutation_rate: f64,
    reinsertion_ratio: f64,
}

impl Default for Parameter {
    fn default() -> Self {
        Self {
            population_size: 100,
            generation_limit: 50000,
            num_individuals_per_parents: 3, // musst be 3 for DifferentialCrossover
            selection_ratio: 0.3,
            crossover_scale: 0.9,
            recombination: 0.2,
            mutation_rate: 0.01, //25,
            reinsertion_ratio: 0.3,
        }
    }
}

type Int = i16;

fn reconstruct_3d_array(arr: &Array2<Int>, shape2: usize) -> Array3<i8> {
    // construct the 3d array
    let mut arr3d = Array3::<i8>::default((arr.shape()[0], arr.shape()[1], shape2));
    for i in 0..arr.shape()[0] {
        for j in 0..arr.shape()[1] {
            let k = arr[[i, j]] as usize;
            if k < shape2 {
                arr3d[[i, j, k]] = 1;
            }
        }
    }
    arr3d
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
        let proj_0 = arr3d.sum_axis(Axis(0)).mapv(|x| x as Int);
        let proj_1 = arr3d.sum_axis(Axis(1)).mapv(|x| x as Int);
        let proj_2 = arr3d.sum_axis(Axis(2)).mapv(|x| x as Int);
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
        let mut arr3d = Array3::<bool>::default(shape);
        for i in 0..shape.0 {
            for j in 0..shape.1 {
                if shape2_mask[[i, j]] != Int::zero() {
                    let k = rng.gen_range(0..shape.2);
                    arr3d[[i, j, k]] = true;
                }
            }
        }

        // convert the index of the 1s along the 2nd axis to their indices
        let arr = Array2::<Int>::from_shape_fn((shape.0, shape.1), |(i, j)| {
            (0..shape.2)
                .position(|k| arr3d[[i, j, k]])
                .unwrap_or(shape.2) as Int
        });

        // compute the projections
        let proj_0 = arr3d
            .mapv(|x| if x { 1 as Int } else { Int::zero() })
            .sum_axis(Axis(0));
        let proj_1 = arr3d
            .mapv(|x| if x { 1 as Int } else { Int::zero() })
            .sum_axis(Axis(1));
        let proj_2 = arr3d
            .mapv(|x| if x { 1 as Int } else { Int::zero() })
            .sum_axis(Axis(2));

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

/// The fitness function for `Genome`s.
#[derive(Clone, Debug)]
struct FitnessCalc {
    target_projections: (Array2<Int>, Array2<Int>, Array2<Int>),
    max_diff: Int,
}

impl FitnessCalc {
    pub fn new(target_projections: (Array2<Int>, Array2<Int>, Array2<Int>)) -> Self {
        let genome_shape = (
            target_projections.1.shape()[0],
            target_projections.0.shape()[0],
            target_projections.0.shape()[1],
        );
        let max_diff = (genome_shape.0 * genome_shape.1 * genome_shape.2 * 2) as Int;
        FitnessCalc {
            target_projections,
            max_diff,
        }
    }
}

impl FitnessFunction<Genome, usize> for FitnessCalc {
    fn fitness_of(&self, genome: &Genome) -> usize {
        // compute the projections of the genome
        let proj_0 = genome.project(0);
        let proj_1 = genome.project(1);
        // let proj_2 = genome.project(2);

        // take the sum of the absolute differences
        // between the projections and the target projections
        let diff_0 = (proj_0 - &self.target_projections.0)
            .mapv(|x| x.abs())
            .sum();
        let diff_1 = (proj_1 - &self.target_projections.1)
            .mapv(|x| x.abs())
            .sum();
        // let diff_2 = (proj_2 - &self.target_projections.2).mapv(|x| abs(x)).sum();
        let diff = diff_0 + diff_1; // + diff_2;

        // subtract from the maximum
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
    recombination: f32,
    crossover_scale: f32,
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
            let a = children.len();
            let (b, c) = match a {
                0 => (1, 2),
                1 => (0, 2),
                2 => (0, 1),
                _ => panic!("Invalid number of parents"),
            };
            let mut genome_arr = parents[a].arr.clone();
            let parent_diff = (parents[b].arr.clone() - parents[c].arr.clone())
                .mapv(|x| ((x as f32) * self.crossover_scale).round() as Int);
            let b_first = genome_arr.clone() + parent_diff;
            let num_indices = genome_arr.len() * self.recombination as usize;
            for _ in 0..num_indices {
                let i = rng.gen_range(0..genome_shape[0]);
                let j = rng.gen_range(0..genome_shape[1]);
                genome_arr[[i, j]] = b_first[[i, j]];
            }

            children.push(Genome::from_arr(genome_arr, genome_shape_2));
        }
        children
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

impl Sub for Genome {
    type Output = Array3<i8>;
    fn sub(self, other: Self) -> Self::Output {
        // reconstruct the 3d array
        let arr3d_1 = reconstruct_3d_array(&self.arr, self.projections.2.shape()[1]);
        let arr3d_2 = reconstruct_3d_array(&other.arr, other.projections.2.shape()[1]);
        (arr3d_1 - arr3d_2).mapv(|x| x.abs())
    }
}

fn new_population(
    size: usize,
    shape: (usize, usize, usize),
    shape2_mask: &Array2<Int>,
    rng: &mut impl Rng,
) -> Population<Genome> {
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
    let shape = (10, 10, 10);
    let rng = &mut rand::thread_rng();
    let p = 0.5;
    // generate a random boolean matrix as a mask for the 3rd axis
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
    let fitness_calc = FitnessCalc::new(ground_truth.projections.clone());

    let params = Parameter::default();

    let initial_population = new_population(
        params.population_size,
        shape,
        &ground_truth.projections.2,
        rng,
    );

    //////////////////////////////////////////////////
    let mut projection_sim = simulate(
        genetic_algorithm()
            .with_evaluation(fitness_calc.clone())
            .with_selection(MaximizeSelector::new(
                params.selection_ratio,
                params.num_individuals_per_parents,
            ))
            .with_crossover(DifferentialCrossover {
                recombination: params.recombination,
                crossover_scale: params.crossover_scale,
            })
            .with_mutation(RandomValueMutator::new(
                params.mutation_rate,
                0 as Int,
                shape.2 as Int,
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

    /////////////////////////////////////////////////
    mem_stats();

    println!("Starting projection optimization with: {:?}", params);

    loop {
        let result = projection_sim.step();
        match result {
            Ok(SimResult::Intermediate(step)) => {
                let evaluated_population = step.result.evaluated_population;
                let best_solution = step.result.best_solution;
                let diff = (ground_truth.clone() - best_solution.solution.genome.clone()).sum()
                    as f32
                    / (shape.0 * shape.1) as f32;
                println!(
                    "Step: generation: {}, average_fitness: {}, \
                    best fitness: {}, real diff: {}",
                    step.iteration,
                    evaluated_population.average_fitness(),
                    best_solution.solution.fitness as f32 / fitness_calc.max_diff as f32,
                    diff,
                );
                mem_stats();
            }
            Ok(SimResult::Final(step, _processing_time, duration, stop_reason)) => {
                let best_solution = step.result.best_solution;
                let diff = (ground_truth - best_solution.solution.genome).sum() as i32;
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
