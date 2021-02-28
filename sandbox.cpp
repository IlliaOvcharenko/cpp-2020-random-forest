#include <iostream>
#include <map>
#include <algorithm>

int main() {
    std::map<int, double> m;
    m[2] = 1.2;
    m[5] = 0.1;
    m[3] = 0.1;
    m[43] = 1.3;
    m[23] = 1.3;


    for (auto& p: m) {
        std::cout << p.first << " " << p.second << std::endl;
    }

    auto max_el = std::max_element(m.begin(), m.end(), [](auto& a, auto& b) { return a.second < b.second; });
    std::cout << "max element" << std::endl;
    std::cout << max_el->first << " " << max_el->second << std::endl;
    return 0;
}
