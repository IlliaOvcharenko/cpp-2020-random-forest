#include <iostream>
#include <map>
#include <algorithm>
#include <functional>
#include <cstdlib>
#include <numeric>
#include <random>


void just_vector_function(std::vector<std::vector<int>>& v) {
    for(auto & i : v) {
        for(int j : i) {
            std::cout << j << " ";
        }
        std::cout << std::endl;
    }
    std::cout << "~~~~~~~~~~~~~~" << std::endl;
}

std::vector<int> generate_subset(int len) {
    std::vector<int> subset;
    for (int i = 0; i < len; ++i) {
        subset.push_back(rand() % len);
    }
    return subset;
}

std::vector<bool> generate_feature_mask(int n_features, int n_leave) {
    std::vector<bool> feature_mask(n_features, false);
    int n_filled = 0;

    while ((n_features - n_filled) > n_leave) {
        std::cout << "n filled: " << n_filled << std::endl;
        int pos = rand() % n_features;
        feature_mask[pos] = true;
        n_filled = std::accumulate(feature_mask.begin(), feature_mask.end(), 0);
    }

    return feature_mask;
}


int main() {
//    std::map<int, double> m;
//    m[2] = 1.2;
//    m[5] = 0.1;
//    m[3] = 0.1;
//    m[43] = 1.3;
//    m[23] = 1.3;
//
//    for (auto& p: m) {
//        std::cout << p.first << " " << p.second << std::endl;
//    }
//
//    auto max_el = std::max_element(m.begin(), m.end(), [](auto& a, auto& b) { return a.second < b.second; });
//    std::cout << "max element" << std::endl;
//    std::cout << max_el->first << " " << max_el->second << std::endl;

//    std::vector<std::vector<int>> v;
//    for (int i = 0; i < 5; ++i) {
//        std::vector<int> tmp;
//        for (int j = 0; j < 4; ++j) {
//            tmp.push_back(j*i);
//        }
//        v.push_back((tmp));
//    }
//
//    just_vector_function(v);

//    std::vector<std::reference_wrapper<std::vector<int>>> v_ref;
//    v_ref.push_back(v[0]);
//    v_ref.push_back(v[2]);
//    for(auto & i : v_ref) {
//        for(int j: i) {
//            std::cout << j << " ";
//        }
//        std::cout << std::endl;
//    }
//    std::cout << "~~~~~~~~~~~~~~" << std::endl;
//    just_vector_function(v_ref);
//    srand(42);
//    std::cout << rand() << " " << RAND_MAX << std::endl;

//    for (auto v: generate_feature_mask(10, 3)) {
//        std::cout << v << std::endl;
//    }

    std::mt19937 gen(42);
    std::uniform_int_distribution<> uid1(0, 10);
    for (int i = 0; i < 10; ++i) {
        std::cout << uid1(gen) << " ";
    }
    std::cout << std::endl;

    std::uniform_int_distribution<> uid2(0, 10);
    for (int i = 0; i < 10; ++i) {
        std::cout << uid2(gen) << " ";
    }
    std::cout << std::endl;
    return 0;
}
