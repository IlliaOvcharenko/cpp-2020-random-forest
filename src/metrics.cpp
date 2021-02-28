#include "metrics.h"

double accuracy_score(target_type& y_true, target_type& y_pred) {
    // TODO add assert in case of different sizes
    int n = y_true.size();
    std::vector<bool> is_equal(n, false);
    for (int i = 0; i < n; ++i) {
        is_equal[i] = (y_true[i] == y_pred[i]);
    }

    int tp = std::accumulate(is_equal.begin(), is_equal.end(), 0);
    return static_cast<double>(tp) / n;
}
