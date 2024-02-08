// import rand crate
use genevo::{
    operator::prelude::*, population::ValueEncodedGenomeBuilder, prelude::*, types::fmt::Display,
};
use rand::prelude::*;

#[derive(Debug)]
struct Parameter {
    population_size: usize,
    generation_limit: u64,
    num_individuals_per_parents: usize,
    selection_ratio: f64,
    num_crossover_points: usize,
    mutation_rate: f64,
    reinsertion_ratio: f64,
}

impl Default for Parameter {
    fn default() -> Self {
        Self {
            population_size: 100,
            generation_limit: 100,
            num_individuals_per_parents: 2,
            selection_ratio: 0.5,
            num_crossover_points: 2,
            mutation_rate: 0.01,
            reinsertion_ratio: 0.1,
        }
    }
}

/// The phenotype
// type Phenotype = Vec<Vec<Vec<u8>>>;

/// The genotype
type Genome = Vec<Vec<Vec<u8>>>;

/// The fitness function for `Genome`s.
#[derive(Clone, Debug)]
struct FitnessCalc {
    target_projections: (Vec<Vec<u8>>, Vec<Vec<u8>>, Vec<Vec<u8>>),
    }

impl FitnessFunction<Genome, usize> for FitnessCalc {
    fn fitness_of(&self, genome: &Genome) -> usize {
        // compute the projections of the genome
        let proj_0 = compute_projection(genome, 0);
        let proj_1 = compute_projection(genome, 1);
        let proj_2 = compute_projection(genome, 2);

        // compare the projections with the target projections
        // TODO: check what is going on here
        self.target_projections.0.iter().zip(proj_0.iter())
            .chain(self.target_projections.1.iter().zip(proj_1.iter()))
            .chain(self.target_projections.2.iter().zip(proj_2.iter()))
            .map(|(t, p)| t.iter().zip(p.iter()).map(|(t, p)| (t - p).pow(2)).sum::<u8>() as usize)
            .sum()
    }

    fn average(&self, fitness_values: &[usize]) -> usize {
        fitness_values.iter().sum::<usize>() / fitness_values.len()
    }

    fn highest_possible_fitness(&self) -> usize {
        100_00
    }

    fn lowest_possible_fitness(&self) -> usize {
        0
    }
}

/// Generate a 3D matrix of a given shape and fill it with 1s with a given probability.
fn generate_3D_matrix(shape: (usize, usize, usize), p: f64) -> Vec<Vec<Vec<u8>>> {
    let mut x = vec![vec![vec![0; shape.2]; shape.1]; shape.0];
    for i in 0..shape.0 {
        for j in 0..shape.1 {
            if rand::random::<f64>() < p {
                let idx = rand::random::<usize>() % shape.2;
                x[i][j][idx] = 1;
            }
        }
    }
    x
}

/// A function to compute the projection of a 3D matrix along a given axis.
fn compute_projection(x: &Vec<Vec<Vec<u8>>>, axis: i32) -> Vec<Vec<u8>> {
    let mut projection = vec![vec![0; x[0].len()]; x.len()];
    for i in 0..x.len() {
        for j in 0..x[0].len() {
            for k in 0..x[0][0].len() {
                if axis == 2 {
                    projection[i][j] += x[i][j][k];
                } else if axis == 1 {
                    projection[i][j] += x[i][k][j];
                } else {
                    projection[i][j] += x[k][i][j];
                }
            }
        }
    }
    projection
}

// Exported function for Python
#[no_mangle]
pub extern "C" fn run_genetic_algorithm() {

    let shape = (20, 20, 100);
    let p = 0.8;
    let x = generate_3D_matrix(shape, p);
    let proj_0 = compute_projection(&x, 0); // copilot suggest using clone?
    let proj_1 = compute_projection(&x, 1);
    let proj_2 = compute_projection(&x, 2);
    let fitness_calc = FitnessCalc {
        target_projections: (proj_0, proj_1, proj_2),
    };

    let params = Parameter::default();

    let initial_population: Population<Genome> = build_population()
        // TODO: create cusom genome builder
        .with_genome_builder(ValueEncodedGenomeBuilder::new("ciao ciao".len(), 32, 126))
        .of_size(params.population_size)
        .uniform_at_random();

    let mut projection_sim = simulate(
        genetic_algorithm()
            .with_evaluation(fitness_calc)
            // TODO: find a good setup
            .with_selection(MaximizeSelector::new(
                params.selection_ratio,
                params.num_individuals_per_parents,
            ))
            .with_crossover(MultiPointCrossBreeder::new(params.num_crossover_points))
            .with_mutation(RandomValueMutator::new(params.mutation_rate, 32, 126))
            .with_reinsertion(ElitistReinserter::new(
                fitness_calc,
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

    println!("Starting projection optimization with: {:?}", params);

    loop {
        let result = projection_sim.step();
        match result {
            Ok(SimResult::Intermediate(step)) => {
                let evaluated_population = step.result.evaluated_population;
                let best_solution = step.result.best_solution;
                println!(
                    "Step: generation: {}, average_fitness: {}, \
                     best fitness: {}, duration: {}, processing_time: {}",
                    step.iteration,
                    evaluated_population.average_fitness(),
                    best_solution.solution.fitness,
                    step.duration.fmt(),
                    step.processing_time.fmt()
                );
                println!("      {}", best_solution.solution.genome.as_text());
                //                println!("| population: [{}]", result.population.iter().map(|g| g.as_text())
                //                    .collect::<Vec<String>>().join("], ["));
            },
            Ok(SimResult::Final(step, processing_time, duration, stop_reason)) => {
                let best_solution = step.result.best_solution;
                println!("{}", stop_reason);
                println!(
                    "Final result after {}: generation: {}, \
                     best solution with fitness {} found in generation {}, processing_time: {}",
                    duration.fmt(),
                    step.iteration,
                    best_solution.solution.fitness,
                    best_solution.generation,
                    processing_time.fmt()
                );
                println!("      {}", best_solution.solution.genome.as_text());
                break;
            },
            Err(error) => {
                println!("{}", error);
                break;
            },
        }
    }
}
