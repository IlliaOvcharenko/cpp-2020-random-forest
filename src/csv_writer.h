#ifndef CPP_2020_RANDOM_FOREST_CSV_WRITER_H
#define CPP_2020_RANDOM_FOREST_CSV_WRITER_H


#include <string>
#include <sstream>
#include <vector>
#include <iostream>
#include <fstream>

class CsvWriter {
    std::ofstream log_file;
public:
    explicit CsvWriter(const std::string& filename);
    void add_line(const std::vector<std::string>& values);
    void close();
};


#endif //CPP_2020_RANDOM_FOREST_CSV_WRITER_H
