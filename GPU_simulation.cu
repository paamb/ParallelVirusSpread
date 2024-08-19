#include <bitset>
#include <iostream>
#include <random>
#include <cstring>
#include <chrono>
#include <utility>
#include <ostream>
#include <vector_types.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas.h>
// For random numbers
#include <curand.h>
#include <curand_kernel.h>

#define NUM_COLS 64
#define INFECTIONRATE 0.1
#define GRID_DIM 64
#define BLOCK_DIM 32

int num_rows, num_healthy, num_infected;
float num_rows_normalizer;


// Init device variables
__device__ int d_num_rows, d_num_healthy, d_num_infected;
__device__ float d_num_rows_normalizer;
__device__ curandState cuda_random_value_state;

using namespace std;

/*

    Project in COSC3500. 
    Virus spread simulation.

    CUDA version
*/
uint64_t create_64_bit_infectionrate_bitmask(){
    uint64_t random_bit_string = 0;

    for (int i = 0; i < NUM_COLS*INFECTIONRATE; i++){
            int mask = 1ULL << i; 
            random_bit_string |= mask;
    }

    for(int i = 0; i < NUM_COLS*INFECTIONRATE; i++){

        uint64_t current_bit = 1LL << i;
    
        int random_index = rand() % (NUM_COLS);
        uint64_t random_bit = 1LL << random_index;


        if (!(random_bit & random_bit_string)){
            random_bit_string |= random_bit;
            uint64_t neg_current_bit = ~current_bit;
            random_bit_string &= neg_current_bit;  
        }


    }
    return random_bit_string;
}

void pretty_print_bit_matrix(uint64_t* grid){
    int total_ones = 0;
    for(int i = 0; i < num_rows; i++){
        bitset<NUM_COLS> bits(grid[i]);
        cout << bits;
        int ones = bits.count();
        cout << "    " << ones << endl;
        total_ones += ones;
    }

    cout << total_ones << endl;
}
/*

@brief Populates the grid. 


Example of what grid could look like after this step:  
00111
00111
00011
00011

The population have not been shuffled in this function

@param grid A pointer to the grid
@param num_bits_to_place The total number of 1's that should be placed on the grid.

*/
__global__ void populate_grid(uint64_t* grid, int num_bits_to_place){
    // Sets the see. Also include the index to not have the same seed every time
    int idx = blockIdx.x*BLOCK_DIM + threadIdx.x;
    int number_of_threads = GRID_DIM * BLOCK_DIM;
    //Initialize the grids to only contain 0's

    // Spread out equal amount of 1's on each row. The bits are not shuffled here.
    int extra_bits = num_bits_to_place % d_num_rows;
    for(int row = idx; row < d_num_rows; row+=number_of_threads){
        int bits_to_be_placed_in_row = num_bits_to_place * d_num_rows_normalizer;

        // For the first extra_bits rows. We want to add a bit.
        // This is different from the other versions as it is modified for CUDA
        if (row < extra_bits ){
            bits_to_be_placed_in_row++;
        } 
        // Set the bits from least significant bit
        uint64_t mask = (1 << bits_to_be_placed_in_row) - 1;
        grid[row] |= mask;
    }
}

/*

@brief Moves the population in the healthy grid. Each bit will either move to an empty spot, or stay where it is.

*/
__device__ void move_healthy(uint64_t* d_healthy_grid, int thread_index, int num_threads, int iteration){
    // Finds a healthy bit at a position. Moves it to another position if that is not empty.
    // Cuda random value generator
    curandState local_state;
    curand_init(thread_index * iteration + thread_index, iteration, 0, &local_state);

    for (int i = thread_index; i < d_num_rows; i+= num_threads){
        int number_of_ones = 0; 
        for (int j = 0; j < NUM_COLS; j++){
            uint64_t current_bit = 1LL << j;
            // If the current bit is 0. Then break. All trailing bits will be 0 aswell no need to move them. 
            if(!(d_healthy_grid[i] & current_bit)){break;}
            int random_index = __double2int_rn(curand_uniform(&local_state)*63.0f);
            // Shift a 1 by a random index. The position the '1'-bit should be moved to.
            uint64_t random_bit = 1LL << random_index;

            // If the position is empty then move the bit from pos = j to pos = random_index.

            if (!(random_bit & d_healthy_grid[i])){
                number_of_ones++;
                d_healthy_grid[i] |= random_bit;
                uint64_t neg_current_bit = ~current_bit;
                d_healthy_grid[i] = d_healthy_grid[i] & neg_current_bit;  
            }
        }
    }
}


/*

@brief Same function as move_healthy, but needs to take account for not moving to same place an healthy person aswell.

Made the extra function to aviod branching (if-statement to check if in infected or healthy grid)

*/
__device__ void move_infected(uint64_t* d_healthy_grid, uint64_t* d_infected_grid, int thread_index, int num_threads, int iteration){

    // Cuda random value generator
    curandState local_state;
    curand_init(thread_index * iteration + thread_index, iteration, 0, &local_state);
    // This does essentially the same as move_healthy, but it needs to check if of healthy individuals aswell.
    for (int i = thread_index; i < d_num_rows; i+=num_threads){
        for (int j = 0; j < NUM_COLS; j++){
            uint64_t current_bit = 1LL << j;
            if(!(d_infected_grid[i] & current_bit)) {break;};

            int random_index = __double2int_rn(curand_uniform(&local_state)*63.0f);
            // Find random index can only be moved inside the coloum (spatial locality).

            // Shift a 1 by a random index. The position the '1'-bit should be moved to.
            uint64_t random_bit = 1LL << random_index;

            // Make sure to not move to a position where a healthy person is
            if (!(random_bit & d_healthy_grid[i] || d_infected_grid[i] & random_bit)){
                d_infected_grid[i] |= random_bit;
                uint64_t neg_current_bit = ~current_bit;
                d_infected_grid[i] = d_infected_grid[i] & neg_current_bit;
            }
        }
    }
}


/**

Infect function. Healthy people will be infected by sick people around

**/
pair<int, int> infect(uint64_t* infected_grid, uint64_t* healthy_grid){
    


    uint64_t* healthy_grid_copy = (uint64_t*) malloc(num_rows* sizeof(uint64_t));
    uint64_t* infected_grid_copy = (uint64_t*) malloc(num_rows* sizeof(uint64_t));

    memcpy(healthy_grid_copy, healthy_grid, num_rows*sizeof(uint64_t));
    memcpy(infected_grid_copy, infected_grid, num_rows*sizeof(uint64_t));



    int num_infected = 0;
    int num_healthy = 0;

    // Random bit string with 
    uint64_t possible_infect_positions = create_64_bit_infectionrate_bitmask();

    for(int i = 0; i < num_rows; i++){
        // Infect vertically
        // Loop through directions up and down
        uint64_t curr_infection_row = infected_grid[i];
        for(int j = i - 1; j <= i + 1; j++){
            
            // Skipping the top, bottom and the same row for vertically infection
            if (j < 0 || j == i || j >= num_rows){ continue;}

            // Find the people that are up or down for a infected_person
            uint64_t infect_vertically = curr_infection_row & healthy_grid[j];

            // Infect with a rate of INFECTIONRATE
            infect_vertically = infect_vertically & possible_infect_positions;
            
            //Remove healthy people from the grid. By doing NAND-ing with the infected uint64
            healthy_grid_copy[j] &= ~infect_vertically;
            
            // Add the infected people to the infected_grid by OR-ing
            infected_grid_copy[j] |= infect_vertically;
        }

        // Infect to the left and right
        uint64_t left_shift = infected_grid[i] << 1;
        uint64_t right_shift = infected_grid[i] >> 1;
        

        // Shift both ways. Equivalent to infecting both right and left.
        // These are the people that can possibly be infected.
        uint64_t possible_infections_horizontally = (left_shift | right_shift) & healthy_grid[i];
        
        uint64_t actual_infections_horizontally = 0;
        actual_infections_horizontally = possible_infections_horizontally & possible_infect_positions; 


        // Remove from healthy
        healthy_grid_copy[i] &= ~actual_infections_horizontally;

        // Add to infected
        infected_grid_copy[i] |= actual_infections_horizontally;

        // Count number of infected people in the row
        bitset<NUM_COLS> infected_bits(infected_grid_copy[i]);
        num_infected += infected_bits.count();

        // Count number of infected people in the row
        bitset<NUM_COLS> healthy_bits(healthy_grid_copy[i]);
        num_healthy += healthy_bits.count();
    }

    // Copy over the two grids.
    memcpy(healthy_grid, healthy_grid_copy, num_rows*sizeof(uint64_t));
    memcpy(infected_grid, infected_grid_copy, num_rows*sizeof(uint64_t));

    free(healthy_grid_copy); 
    free(infected_grid_copy);

    return make_pair(num_infected, num_healthy);

}

// Set global HOST parameters
void set_global_parameters(){
    num_healthy = num_rows*NUM_COLS*0.2;
    num_infected = num_rows;
    num_rows_normalizer = 1.0/num_rows;
}

// Set global DEVICE parameters
void set_cuda_parameters(){
    cudaMemcpyToSymbol(d_num_rows, &num_rows, sizeof(int), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(d_num_rows_normalizer, &num_rows_normalizer, sizeof(int), 0, cudaMemcpyHostToDevice);
}

__global__ void move(uint64_t *d_healthy_grid, uint64_t *d_infected_grid, int iteration){

    int idx = blockIdx.x*BLOCK_DIM + threadIdx.x;
    int number_of_threads = GRID_DIM * BLOCK_DIM;
    move_healthy(d_healthy_grid, idx, number_of_threads, iteration);
    move_infected(d_healthy_grid, d_infected_grid, idx, number_of_threads, iteration);
}


int main(int argc, char *argv[]){

    if (argc > 1) {
        num_rows = atoi(argv[1]);
    } else {
        num_rows = 8;
    };


    cout << num_rows << endl; 

    set_global_parameters();
    set_cuda_parameters();
    cudaDeviceSynchronize();

    int healthy_people = num_healthy;
    int infected_people = num_infected;

    // Define the two grids. Each spot is represented as bits.
    // For the healthy_grid: HEALTHY = 1, EMPTY = 0
    uint64_t* healthy_grid = new uint64_t[num_rows];
    // For the infected_grid: INFECTED = 1, EMPTY = 0
    uint64_t* infected_grid = new uint64_t[num_rows];

    // Define the device grids
    uint64_t *d_healthy_grid, *d_infected_grid;

    // Allocate memory for the device healthy and infected matrices
    cudaMalloc((void**) &d_healthy_grid, num_rows*sizeof(uint64_t));
    cudaMalloc((void**) &d_infected_grid, num_rows*sizeof(uint64_t));

    // Define copy_device grids. Used in infect function.
    uint64_t *d_healthy_grid_copy, *d_infected_grid_copy;
    cudaMalloc((void**) &d_healthy_grid_copy, num_rows*sizeof(uint64_t));
    cudaMalloc((void**) &d_infected_grid_copy, num_rows*sizeof(uint64_t));


    cout << "Infected at start: " << infected_people << endl;
    cout << "Healthy  at start: " << healthy_people << endl;
    cout << endl;
    cudaMemcpyToSymbol(d_num_healthy, &healthy_people, sizeof(healthy_people), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(d_num_infected, &infected_people, sizeof(infected_people), 0, cudaMemcpyHostToDevice);

    auto start = std::chrono::high_resolution_clock::now();
    for(int i = 0; i < 100; i++){
        cudaMemset(d_healthy_grid, 0, num_rows * sizeof(uint64_t));
        cudaMemset(d_infected_grid, 0, num_rows * sizeof(uint64_t));
        
        // Populate the grids on GPU
        populate_grid<<<GRID_DIM,BLOCK_DIM>>>(d_healthy_grid, healthy_people);
        populate_grid<<<GRID_DIM,BLOCK_DIM>>>(d_infected_grid, infected_people);

        // Move both infected and healthy people on GPU
        move<<<GRID_DIM, BLOCK_DIM>>>(d_healthy_grid, d_infected_grid, i);
        cudaMemcpy(healthy_grid, d_healthy_grid, num_rows*sizeof(uint64_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(infected_grid, d_infected_grid, num_rows*sizeof(uint64_t), cudaMemcpyDeviceToHost);

        // Infect and find number of infected and healthy people on CPU
        pair<int, int> res = infect(infected_grid, healthy_grid);
        infected_people = res.first;
        healthy_people = res.second;
        cudaMemcpyToSymbol(d_num_healthy, &healthy_people, sizeof(healthy_people), 0, cudaMemcpyHostToDevice);
        cudaMemcpyToSymbol(d_num_infected, &infected_people, sizeof(infected_people), 0, cudaMemcpyHostToDevice);
    }
    

    // Copy healthy people from GPU to print
    cudaMemcpyFromSymbol(&healthy_people, d_num_healthy, sizeof(healthy_people), 0, cudaMemcpyDeviceToHost);
    cudaMemcpyFromSymbol(&infected_people, d_num_infected, sizeof(infected_people), 0, cudaMemcpyDeviceToHost);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end-start);
    cout << "Duration: " << duration.count() << endl;

    cout << "Infected at end: " << infected_people << endl;
    cout << "Healthy  at end: " << healthy_people << endl;
        int sum_inf_health = infected_people + healthy_people;
    cout << "Infected + healthy " <<sum_inf_health << endl;

    
    delete[] healthy_grid;
    delete[] infected_grid;
    cudaFree(d_healthy_grid);
    cudaFree(d_infected_grid);
    return 0;
}