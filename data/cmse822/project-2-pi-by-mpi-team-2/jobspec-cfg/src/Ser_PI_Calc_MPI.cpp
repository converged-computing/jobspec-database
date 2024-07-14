#include <iostream>
#include <fstream>
#include <cstdlib>
#include <ctime>
#include <random>
#include <string>
#include "mpi.h"

using namespace std;

// These parameters would be input in command line.
// // Define the main inputs to the code: number of darts, and rounds of play.
// int Darts = 10000000;           // number of darts used in each round.
// int Rounds = 100;           // total number of rounds.

// ==================================================================================================
// ==================================================================================================
// ==================================================================================================

double ThrowDarts(int darts, int seed_value) {
    double x_coord, y_coord, pi, r;
    int score = 0;

    // Creating the random generator.
    mt19937 gen(seed_value);                          // Seed the generator
    uniform_real_distribution<> distr(0.0, 1.0);        // Define the range

    // Start simulation of dart thorwning.
    for (int n = 1; n <= darts; n++) {
        x_coord = distr(gen);           // Generate a random number [0.0, 1.0)
        y_coord = distr(gen);           // Generate another random number [0.0, 1.0]
        // Check if the dart hits inside the circle.
        if ((x_coord * x_coord + y_coord * y_coord) <= 1.0)
            score++;
    }

    pi = 4.0 * score / darts;
    return pi;
}
// ==================================================================================================
// ==================================================================================================
// ==================================================================================================

int main(int argc, char *argv[]) {
    double Pi;              // average of pi after "darts" is thrown in a given round.
    double AvgPi;           // average pi value for all rounds.
    double TotalPi, TotalTime;
    int numtasks, rank;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Check for sufficient command line arguments. Two parameters of "Darts" and "Rounds" should be 
    //  an input from the command line.
    if (argc != 4) {
        if (rank == 0) {
            cout << "Usage: " << argv[0] << " <Number of Darts> <Number of Rounds> <output file name>" << endl;
        }
        MPI_Finalize();
        return 1;
    }

    // Convert command line arguments to integers for "Darts" and "Roudns" parameters.
    int Darts = atoi(argv[1]);
    int Rounds = atoi(argv[2]);
    string out_file_name = argv[3];

    // Get the start time in first process (rank 0 only use this variable.).
    double StartTime = MPI_Wtime();

    if (rank == 0) {
        cout << "Starting MPI version of pi calculation using dartboard algorithm... " << 
            "(Darts: " << Darts << ", Rounds: " << Rounds << ")" << endl;
    }

    AvgPi = 0.0;
    random_device rd;                           // Obtain a random number from hardware
    int seed_value = rd() + (rank * 100);       // Make seed value randomized for each process.

    int RoundsPerProcess = (Rounds / numtasks) + 1;     // Number of rounds each process should run.
    for (int i = 0; i < RoundsPerProcess; i++) {
        Pi = ThrowDarts(Darts, seed_value);
        AvgPi += Pi;
    }
    AvgPi /= RoundsPerProcess;

    // Gather all the average Pi values from each process and compute the total average
    MPI_Reduce(&AvgPi, &TotalPi, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        TotalPi /= numtasks;                        // Calculate the average Pi value.F
        double EndTime = MPI_Wtime();               // Get the wall clock for the end time.
        double ElapsedTime = EndTime - StartTime;   // Calculate the total time, based on rank 0.
        cout << "\tAfter " << (Darts * Rounds) << " throws, average value of pi = " << TotalPi << endl;
        cout << "\tReal value of PI: 3.1415926535897" << endl;
        cout << "\tTotal Run time: " << ElapsedTime << " seconds!" << endl;

        // Save the results in a text file.
        ofstream myfile(out_file_name, ios::app);     // Open the results file in append mode.
        if (myfile.is_open()) {
            myfile << Darts << "," << Rounds << "," << TotalPi << "," << ElapsedTime << "," << numtasks << "\n";
            myfile.close();
        } else {
            cerr << "Unable to open file for writing." << endl;
            return 1;
        }
    }

    MPI_Finalize();

    return 0;
}
// ==================================================================================================
// ==================================================================================================
// ==================================================================================================



