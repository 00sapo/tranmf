// import rand crate
use rand;
use genetic_algorithm::strategy::evolve::prelude::*;


// Define the search goal
#[derive(Clone, Debug)]
struct CountTrue;
impl Fitness for CountTrue {
    type Genotype = BinaryGenotype;
    fn calculate_for_chromosome(&mut self, chromosome: &Chromosome<Self::Genotype>) -> Option<FitnessValue> {
        Some(chromosome.genes.iter().filter(|&value| *value).count() as FitnessValue)
    }
}

// Exported function for Python
#[no_mangle]
pub extern "C" fn run_genetic_algorithm() {
    // Define the search space
    let genotype = BinaryGenotype::builder()
        .with_genes_size(100) //  100 genes per chromosome
        .build()
        .unwrap();
    // Initialize the randomness provider
    let mut rng = rand::thread_rng();

    // Set up the genetic algorithm strategy
    let evolve = Evolve::builder()
        .with_genotype(genotype)
        .with_target_population_size(100) // evolve with  100 chromosomes
        .with_target_fitness_score(100)   // goal is  100 times true in the best chromosome
        .with_fitness(CountTrue)          // count the number of true values in the chromosomes
        .with_crossover(CrossoverUniform::new(true)) // crossover all individual genes between  2 chromosomes for offspring
        .with_mutate(MutateOnce::new(0.2))    // mutate a single gene with a  20% probability per chromosome
        .with_compete(CompeteElite::new())    // sort the chromosomes by fitness to determine crossover order
        .with_extension(ExtensionNoop::new()) // extension step, disabled
        .call(&mut rng)
        .unwrap();

    // Return the result as a string
    println!("{}", evolve);
}
