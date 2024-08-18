#include <bitset>
#include <iostream>
#include <random>
#include <cstring>
#include <chrono>
#include <utility>
#include <ostream>
#include <omp.h>
#include <mpi.h>

#define NUM_COLS 64
#define INFECTIONRATE 0.1

int num_rows, num_healthy, num_infected;
int step_size = 4;
float num_rows_normalizer;


using namespace std;

/*

    Project in COSC3500. 
    Virus spread simulation.

*/

void pretty_print_bit_matrix(uint64_t* grid, int matrix_size){
    int total_ones = 0;
    for(int i = 0; i < matrix_size; i++){
        bitset<NUM_COLS> bits(grid[i]);
        cout << bits;
        int ones = bits.count();
        cout << "    " << ones << endl;
        total_ones += ones;
    }

    cout << total_ones << endl;
}

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
void populate_grid(uint64_t* grid, int num_bits_to_place){
    //Initialize the grids to only contain 0's
    memset(grid, 0, num_rows * sizeof(uint64_t));

    // Spread out equal amount of 1's on each row. The bits are not shuffled here.
    int extra_bits = num_bits_to_place % num_rows;

    for(int row = 0; row < num_rows; row++){
        int bits_to_be_placed_in_row = num_bits_to_place * num_rows_normalizer;

        if(extra_bits > 0){
            bits_to_be_placed_in_row++;
            extra_bits--;
        }

        // Set the bits from least significant bit
        uint64_t mask = (1 << bits_to_be_placed_in_row) - 1;
        grid[row] |= mask;
    }
}





void compute_section_move_infected(uint64_t* healthy_grid, uint64_t* infected_grid, int thread_id, int iteration){
    // Two nodes and 4 threads each. Have to split in 8 blocks
    int start_index = num_rows / 8 * thread_id;
    int end_index = start_index + num_rows / 8;
    for (int i = start_index; i < end_index; i++){
        unsigned int seed = (unsigned int) time(0) + 123 + thread_id * i + iteration;
        for (int j = 0; j < NUM_COLS; j++){

            uint64_t current_bit = 1LL << j;
            if(!(infected_grid[i] & current_bit)) {break;};

            // Find random index can only be moved inside the coloum (spatial locality).
            int random_index = rand_r(&seed) % (NUM_COLS);

            // Shift a 1 by a random index. The position the '1'-bit should be moved to.
            uint64_t random_bit = 1LL << random_index;

            // Make sure to not move to a position where a healthy person is
            if (!(random_bit & healthy_grid[i] || infected_grid[i] & random_bit)){
                infected_grid[i] |= random_bit;
                uint64_t neg_current_bit = ~current_bit;
                infected_grid[i] = infected_grid[i] & neg_current_bit;
            }
        }
    }
}

void compute_section_move_healthy(uint64_t* half_healthy_grid, int thread_id, int iteration){
    // Two nodes and 4 threads each. Have to split in 8 blocks
    int start_index = num_rows / 8 * thread_id;
    int end_index = start_index + num_rows / 8;
    for (int i = start_index; i < end_index; i++){
        unsigned int seed = (unsigned int) time(0) + 456 + thread_id * i + iteration;
        for (int j = 0; j < NUM_COLS; j++){
            uint64_t current_bit = 1LL << j;

            // If the current bit is 0. Then break. All trailing bits will be 0 aswell no need to move them. 
            // See populate_grid() - function.
            if(!(half_healthy_grid[i] & current_bit)) {break;};

            // Find random index can only be moved inside the coloum (spatial locality).
            int random_index = rand_r(&seed) % (NUM_COLS);
            
            // Shift a 1 by a random index. The position the '1'-bit should be moved to.
            uint64_t random_bit = 1LL << random_index;

            // If the position is empty then move the bit from pos = j to pos = random_index.
            if (!(random_bit & half_healthy_grid[i])){
                half_healthy_grid[i] |= random_bit;
                uint64_t neg_current_bit = ~current_bit;
                half_healthy_grid[i] = half_healthy_grid[i] & neg_current_bit;  
            }
        }
    }
}


/*

@brief Moves the population in the healthy grid. Each bit will either move to an empty spot, or stay where it is.

*/
void move_healthy(uint64_t* healthy_grid, int node_rank, int iteration){
    // Finds a healthy bit at a position. Moves it to another position if that is not empty.

    // Node 0 will move upper half of healthy_matrix. Node 1 will move lower half
    uint64_t* half_healthy_grid = (uint64_t*) malloc(num_rows/2 * sizeof(uint64_t)); 
    uint64_t* healthy_grid_start = healthy_grid + num_rows/2 * node_rank;
    memcpy(half_healthy_grid, healthy_grid_start, num_rows*sizeof(uint64_t)/2);

    #pragma omp parallel shared(half_healthy_grid, node_rank, iteration)
    {
    #pragma omp sections
        {

    #pragma omp section
        {

            compute_section_move_healthy(half_healthy_grid, 0, iteration);
        }

    #pragma omp section
        {
            compute_section_move_healthy(half_healthy_grid, 1, iteration);
        }
    #pragma omp section
        {
            compute_section_move_healthy(half_healthy_grid, 2, iteration);
        }
    #pragma omp section
        {
            compute_section_move_healthy(half_healthy_grid, 3, iteration);
        }
        }
    }
    memcpy(healthy_grid_start, half_healthy_grid, num_rows*sizeof(uint64_t)/2);
    free(half_healthy_grid);
}



void move_infected(uint64_t* healthy_grid, uint64_t* infected_grid, int node_rank, int iteration){
    // Node 0 will move upper half of infection matrix. Node 1 will move lower half.
    uint64_t* half_healthy_grid = (uint64_t*) malloc(num_rows/2 * sizeof(uint64_t)); 
    uint64_t* half_infected_grid = (uint64_t*) malloc(num_rows/2 * sizeof(uint64_t)); 

    // Will start from healthy_grid when node_rank = 0
    uint64_t* healthy_grid_start = healthy_grid + num_rows/2 * node_rank;
    uint64_t* infected_grid_start = infected_grid + num_rows/2 * node_rank;

    memcpy(half_healthy_grid, healthy_grid_start, num_rows*sizeof(uint64_t)/2);
    memcpy(half_infected_grid, infected_grid_start, num_rows*sizeof(uint64_t)/2);
#pragma omp parallel shared(half_healthy_grid, half_infected_grid, node_rank, iteration)
  {

#pragma omp sections
    {

#pragma omp section
      {
        compute_section_move_infected(half_healthy_grid, half_infected_grid, 0, iteration);
      }

#pragma omp section
      {
        compute_section_move_infected(half_healthy_grid, half_infected_grid, 1, iteration);
      }
#pragma omp section
      {
        compute_section_move_infected(half_healthy_grid, half_infected_grid, 2, iteration);
      }
#pragma omp section
      {
        compute_section_move_infected(half_healthy_grid, half_infected_grid, 3, iteration);
      }
    }
  }
  memcpy(infected_grid_start, half_infected_grid, num_rows*sizeof(uint64_t)/2);
  free(half_healthy_grid);
  free(half_infected_grid);
}


/**
    Same as for the single thread simulation
**/
pair<int, int> infect(uint64_t* infected_grid, uint64_t* healthy_grid, int node_rank){
    
    // Copy the grids to not affect original grids while infecting.
    uint64_t* healthy_grid_copy = (uint64_t*) malloc(num_rows* sizeof(uint64_t));
    uint64_t* infected_grid_copy = (uint64_t*) malloc(num_rows* sizeof(uint64_t));

    memcpy(healthy_grid_copy, healthy_grid, num_rows*sizeof(uint64_t));
    memcpy(infected_grid_copy, infected_grid, num_rows*sizeof(uint64_t));


    int num_infected = 0;
    int num_healthy = 0;

    // Random bit string with 
    uint64_t possible_infect_positions = create_64_bit_infectionrate_bitmask();

    int start_index = num_rows/2 * node_rank;
    int end_index = start_index + num_rows/2;

    for(int i = start_index; i < end_index; i++){
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


void set_global_parameters(){
    num_healthy = num_rows*NUM_COLS*0.2;
    num_infected = num_rows;
    num_rows_normalizer = 1.0/num_rows;
}

int main(int argc, char *argv[]){

    MPI_Init(&argc, &argv);

    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    if (argc > 1) {
        num_rows = atoi(argv[1]);
    } else {
        num_rows = 8;
    };

    set_global_parameters();

    int healthy_people = num_healthy;
    int infected_people = num_infected;


    // Define the two grids. Each spot is represented as bits.
    // For the healthy_grid: HEALTHY = 1, EMPTY = 0
    uint64_t* healthy_grid = new uint64_t[num_rows];

    // For the infected_grid: INFECTED = 1, EMPTY = 0
    uint64_t* infected_grid = new uint64_t[num_rows];
    // uint64_t* infected_grid = new uint64_t[num_rows];



    cout << "Infected at start: " << infected_people << endl;
    cout << "Healthy  at start: " << healthy_people << endl;
    cout << endl;

    auto start = std::chrono::high_resolution_clock::now();




    for (int i = 0; i < 100; i++){
        // Set barrier to sync before move step
        MPI_Barrier(MPI_COMM_WORLD);

        // Populate both grids with ones. First step of shuffeling the grid.
        populate_grid(healthy_grid, healthy_people);
        populate_grid(infected_grid, infected_people);

        move_healthy(healthy_grid, my_rank, i);
        move_infected(healthy_grid, infected_grid, my_rank, i);


        // Infect the healthy people. Update the number of sick and healthy people
        int half_infected_people, half_healthy_people; 

        pair<int, int> res = infect(infected_grid, healthy_grid, my_rank);
        half_infected_people = res.first;
        half_healthy_people = res.second;

        int healthy_people_from_other_node, infected_people_from_other_node; 

        // This is where the synchronization happens.
        // Sending the calculated healthy and infected people from both halves to add togheter
        // The grids will now be ready to be populated before next iteration
        MPI_Barrier(MPI_COMM_WORLD);
        if (my_rank == 0){
            MPI_Send(&half_healthy_people, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
            MPI_Send(&half_infected_people, 1, MPI_INT, 1, 1, MPI_COMM_WORLD);
            MPI_Recv(&healthy_people_from_other_node, 1, MPI_INT, 1, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(&infected_people_from_other_node, 1, MPI_INT, 1, 4, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        else{
            MPI_Recv(&healthy_people_from_other_node, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(&infected_people_from_other_node, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Send(&half_healthy_people, 1, MPI_INT, 0, 3, MPI_COMM_WORLD);
            MPI_Send(&half_infected_people, 1, MPI_INT, 0, 4, MPI_COMM_WORLD);
        }
        infected_people = half_infected_people + infected_people_from_other_node;
        healthy_people = half_healthy_people + healthy_people_from_other_node;
    }


    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end-start);
    cout << "Process:  " << my_rank << endl;
    cout << "Duration: " << duration.count() << endl;

    cout << "Infected at end: " << infected_people << endl;
    cout << "Healthy  at end: " << healthy_people << endl;
    int sum_inf_health = infected_people + healthy_people;
    cout << "Infected + healthy " <<sum_inf_health << endl;

    int rc = MPI_Finalize(); 

    delete[]healthy_grid;
    delete[] infected_grid;
    return 0;
}