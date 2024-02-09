// import rand crate
use genevo::{
    operator::prelude::*, mutation::value::RandomGenomeMutation, prelude::*, types::fmt::Display, operator::CrossoverOp, genetic::Parents
};
use ndarray::prelude::*;
// import trait New


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
            population_size: 1000,
            generation_limit: 100000,
            num_individuals_per_parents: 2,
            selection_ratio: 0.5,
            // num_crossover_points: 2,
            mutation_rate: 0.1,
            reinsertion_ratio: 0.1,
        }
    }
}


/// The genotype
#[derive(PartialEq, Clone, Debug)]
struct Genome {
    arr: Array3<bool>
}

impl Genome{
    pub fn new(shape: &[usize]) -> Self {
        Genome{arr: Array3::<bool>::default((shape[0], shape[1], shape[2]))}
    }

    pub fn project(&self, axis: usize) -> Array2<i32> {
        // convert the bool array to an array of 1 and 0 of type i32
        let arr = self.arr.mapv(|x| if x {1} else {0});
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
        let diff = diff_0 as usize + diff_1 as usize + diff_2 as usize;

        // subtract from the maximum and divide by the maximum
        let max_diff = self.target_projections.0.len() + self.target_projections.1.len() + self.target_projections.2.len();
        (max_diff - diff) / max_diff * 100
    }

    fn average(&self, fitness_values: &[usize]) -> usize {
        fitness_values.iter().sum::<usize>() / fitness_values.len()
    }

    fn highest_possible_fitness(&self) -> usize {
        100
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

impl RandomGenomeMutation for Genome {
    type Dna = bool;
    fn mutate_genome<R>(
            genome: Self, 
            mutation_rate: f64, 
            _min_value: &<Self as Genotype>::Dna, 
            _max_value: &<Self as Genotype>::Dna, 
            _rng: &mut R
        ) -> Self
        where
            R: Rng + Sized
    {
        let mut mutated_genome = genome.clone();
        let shape = genome.arr.shape();
        for i in 0..shape[0] {
            for j in 0..shape[1] {
                for k in 0..shape[2] {
                    if rand::random::<f64>() < mutation_rate {
                        mutated_genome.arr[[i, j, k]] = !mutated_genome.arr[[i, j, k]];
                    }
                }
            }
        }
        mutated_genome
    }
}


/// Custom genome builder
struct MyGenomeBuilder
    {
    shape: (usize, usize, usize),
        p: f64,
    }

impl GenomeBuilder<Genome> for MyGenomeBuilder {
    fn build_genome<R>(&self, _: usize, _rng: &mut R) -> Genome
        where
            R: Rng + Sized {
        generate_genome(self.shape, self.p)
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
                let mut genome = Genome::new(genome_shape);
                // for each cell on the 2nd dimension (a column)
                for i in 0..genome_shape[0] {
                    for j in 0..genome_shape[1] {
                        // pick the value of a randomly chosen parent
                        let random = rng.gen_range(0..num_parents);
                        let value = parents[random].arr.slice(s![i, j, ..]);
                        genome.arr.slice_mut(s![i, j, ..]).assign(&value);
                    }
                }
                children.push(genome);
            }
            children
        }
    }

// Exported function for Python
#[no_mangle]
pub extern "C" fn run_genetic_algorithm() {

    let shape = (20, 20, 100);
    let p = 0.8;
    let x = generate_genome(shape, p);
    let proj_0 = x.project(0);
    let proj_1 = x.project(1);
    let proj_2 = x.project(2);
    let fitness_calc = FitnessCalc {
        target_projections: (proj_0, proj_1, proj_2),
    };

    let params = Parameter::default();

    let initial_population: Population<Genome> = build_population()
        .with_genome_builder(MyGenomeBuilder { shape, p })
        .of_size(params.population_size)
        .uniform_at_random();

    let mut projection_sim = simulate(
        // TODO: find a good setup
        genetic_algorithm()
            .with_evaluation(fitness_calc.clone())
            .with_selection(MaximizeSelector::new(
                params.selection_ratio,
                params.num_individuals_per_parents,
            ))
            .with_crossover(UniformCrossBreeder::default())
            .with_mutation(RandomValueMutator::new(params.mutation_rate, false, true))
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
                break;
            },
            Err(error) => {
                println!("{}", error);
                break;
            },
        }
    }
}
