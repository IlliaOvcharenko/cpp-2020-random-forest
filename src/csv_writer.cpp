#include "csv_writer.h"

CsvWriter::CsvWriter(const std::string& filename) {
    log_file.open(filename);
}

void CsvWriter::add_line(const std::vector<std::string>& values) {
    std::stringstream line;
    for (int i = 0; i < values.size()-1; ++i) {
        line << values[i] << ",";
    }
    line << values.back() << "\n";
    log_file << line.str();
}

void CsvWriter::close() {
    log_file.close();
}