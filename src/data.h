#ifndef CPP_2020_RANDOM_FOREST_DATA_H
#define CPP_2020_RANDOM_FOREST_DATA_H

#include <iostream>
#include <fstream>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <map>
#include <algorithm>
#include <memory>
#include <cmath>
#include <numeric>

#include <boost/algorithm/string.hpp>


typedef std::vector<bool> feature_type;
typedef std::vector<feature_type> feature_matrix_type;
typedef std::vector<int> target_type;

struct dataset {
    feature_matrix_type X;
    target_type y;
};

void read_dataset(const std::string& filename, dataset& ds);

#endif //CPP_2020_RANDOM_FOREST_DATA_H
