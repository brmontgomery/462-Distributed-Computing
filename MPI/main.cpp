#include <stdio.h>
#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <algorithm>
#include <sstream>
#include <mpi.h>

int main(int argc, char* argv[]) {
    //program vars
    std::string matrixFile = "";
    int idx = 0;
    int rows = 0;
    int columns = 0;
    //initialize vector of numbers to convert strings into
    std::vector<int> numbers;

    //MPI vars
    int rank;
    int size;
    int rootRank = 0;
    
    //Get the matrix File name
    for (idx = 1; idx < argc; idx++) {
        if (strcmp(argv[idx], "-mFile") == 0) {
            if (idx + 1 < argc) {
                matrixFile = std::string(argv[idx + 1]);
            }
            else {
                printf("Error: No file following the -mFile argument!\n");
                return -1;
            }
        }
    }

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    std::vector<int> localNumbers;
    int portion = 0;
    int cols = 0;
    int rws = 0;

    if (rank == rootRank) {
        std::vector<std::string> rowVec;

        if (matrixFile != "") {
            //load the file into a vector of strings to be parsed once the array has been initilized.
            std::ifstream in(matrixFile);

            rowVec.push_back("");
            while (std::getline(in, rowVec[rows])) {
                size_t n = std::count(rowVec[rows].begin(), rowVec[rows].end(), ',') + 1;
                if (n > columns) {
                    columns = n;
                }
                ++rows;
                rowVec.push_back("");
            }
            rowVec.pop_back();

            rws = rowVec.size();

            in.close();
        }
        else {
            printf("Error: No matrix file was entered with the -mFile argument!\n");
        }

        //populate vector
        for (int i = 0; i < rowVec.size(); i++) {
            std::stringstream ss(rowVec[i]);
            size_t n = std::count(rowVec[i].begin(), rowVec[i].end(), ',') + 1;

            for (int j = 0; j < columns; j++) {
                if (j < n) {
                    std::string substr;
                    getline(ss, substr, ',');
                    substr.erase(std::remove_if(substr.begin(), substr.end(), [](char c) { return !std::isdigit(c); }), substr.end());
                    numbers.push_back(std::stoi(substr));
                }
                else {
                    numbers.push_back(0);
                }
            }
        }

        if (size > 1) {
            portion = numbers.size() / int(size);
        }
    }

    if (rows != 0) {
        cols = numbers.size() / rows;
    }

    //make sure all processors know what the portion size is
    MPI_Bcast(&portion, 1, MPI_INT, rootRank, MPI_COMM_WORLD);
    MPI_Bcast(&cols, 1, MPI_INT, rootRank, MPI_COMM_WORLD);
    localNumbers.resize(portion);

    //scatter the data in the array across the rest of the processors
    MPI_Scatter(numbers.data(), portion, MPI_INT, localNumbers.data(), portion, MPI_INT, rootRank, MPI_COMM_WORLD);
    
    MPI_Barrier(MPI_COMM_WORLD);

    //standard multi-process processing
    std::vector<int> solution;
    if (size > 1) {
        for (int i = 0; i < portion; i++) {
            if (localNumbers[i] != 0) {
                solution.push_back(localNumbers[i]);
                solution.push_back((rank * portion + i) % cols);
                solution.push_back((rank * portion + i) / cols);
            }
        }
    }

    if (rank == rootRank) {
        //processing of any leftovers
        if (size > 1) {
            if (numbers.size() % int(size) != 0) {
                for (int i = 0; i < numbers.size() % int(size); i++) {
                    if (numbers[portion * int(size)] != 0) {
                        solution.push_back(numbers[portion * int(size)]);
                        solution.push_back((size* portion + i) % cols);
                        solution.push_back((size* portion + i) / cols);
                    }
                }
            }
        }
        else {
            if (numbers.size() != 0) {
                for (int i = 0; i < numbers.size(); i++) {
                    if (numbers[i] != 0) {
                        solution.push_back(numbers[i]);
                        solution.push_back(i % cols);
                        solution.push_back(i / cols);
                    }
                }
            }
        }
    }

    //for purposes of demonstrating multiple processes
    std::cout << "Rank: " << rank << " Solution: ";
    for (int i = 0; i < solution.size(); i += 3) {
        std::cout << "(value = " << solution[i] << ", column = " << solution[i + 1] + 1 << ", row = " << solution[i + 2] + 1 << "), ";
    }
    std::cout << std::endl;

    //Gather vars
    std::vector<int> gatherSizes;
    std::vector<int> displacements;
    int solSize = solution.size();

    //prepare the rootRank to Gather
    if (rank == rootRank) {
        //get the sizes of the buffers we're gathering... they may differ
        for (int i = 0; i < size; i++) {
            if (i != rootRank) {
                gatherSizes.push_back(0);
                MPI_Recv(&gatherSizes[i], 1, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
            else {
                gatherSizes.push_back(solution.size());
            }
        }
        //get the according displacements
        int sum = 0;
        std::cout << std::endl;
        for (int i = 0; i < gatherSizes.size(); i++) {
            displacements.push_back(sum);
            sum += gatherSizes[i];
        }

        //gather
        std::vector<int> finalAnswer;
        finalAnswer.resize(sum);
        MPI_Gatherv(solution.data(), solution.size(), MPI_INT, finalAnswer.data(), gatherSizes.data(), displacements.data(), MPI_INT, rootRank, MPI_COMM_WORLD);

        //print finalAnswer
        std::cout << "Final Solution: ";
        for (int i = 0; i < finalAnswer.size(); i += 3) {
            std::cout << "(value = " << finalAnswer[i] << ", column = " << finalAnswer[i + 1] + 1 << ", row = " << finalAnswer[i + 2] + 1 << "), ";
        }
        std::cout << std::endl;
    }
    else {
        //send the sizes of the solution buffer from each process
        MPI_Send(&solSize, 1, MPI_INT, rootRank, 0, MPI_COMM_WORLD);
        //then gather the results into one buffer
        MPI_Gatherv(solution.data(), solution.size(), MPI_INT, NULL, NULL, NULL, MPI_INT, rootRank, MPI_COMM_WORLD);
    }
    

    MPI_Finalize();
    return 0;
}