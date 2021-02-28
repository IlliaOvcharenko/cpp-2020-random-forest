#include "data.h"

void read_dataset(const std::string& filename, dataset& ds) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cout << "file not opened" << std::endl;
    }

    std::string line;

    getline(file,line);
    std::vector<std::string> columns;
    boost::algorithm::split(columns, line, boost::is_any_of(","));

    while (getline(file,line)) {
        std::vector<std::string> values_str;
        std::vector<bool> values_bool;
        boost::algorithm::split(values_str, line, boost::is_any_of(","));

        ds.y.push_back(std::stoi(values_str.back()));
        values_str.pop_back();

        for (const auto& val: values_str) {
            values_bool.push_back(std::stoi(val));
        }
        ds.X.push_back(values_bool);
    }
    file.close();
}
