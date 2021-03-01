#include <iostream>
#include <string>

#include "src/data.h"
#include "src/decision_tree.h"
#include "src/random_forest.h"
#include "src/metrics.h"
#include "src/time_measurement.h"



int main() {
    std::string train_dataset_filename = "../data/train.csv";
    std::string val_dataset_filename = "../data/val.csv";
    dataset train_dataset;
    dataset val_dataset;

    read_dataset(train_dataset_filename, train_dataset);
    read_dataset(val_dataset_filename, val_dataset);

    auto start_time = get_current_time_fenced();
//    Tree model(0.1, 10, true);
    RandomForest model(20, 0.1, 20, 42, 4);
    model.fit(train_dataset);

    auto train_preds = model.predict(train_dataset.X);
    auto train_score = accuracy_score(train_dataset.y, train_preds);

    auto val_preds = model.predict(val_dataset.X);
    auto val_score = accuracy_score(val_dataset.y, val_preds);

    auto finish_time = get_current_time_fenced();
    auto total_time = finish_time - start_time;

    std::cout << "total time, s: " << to_sec(total_time) << std::endl;
    std::cout << "train score: " << train_score << \
          ", validation score: " << val_score << std::endl;



    return 0;
}
