#include <iostream>
#include <string>

#include "src/data.h"
#include "src/random_forest.h"
#include "src/metrics.h"
#include "src/time_measurement.h"
#include "src/csv_writer.h"


int main() {
    std::string train_dataset_filename = "../data/train.csv";
    std::string val_dataset_filename = "../data/val.csv";
    dataset train_dataset;
    dataset val_dataset;

    read_dataset(train_dataset_filename, train_dataset);
    read_dataset(val_dataset_filename, val_dataset);

    std::string log_filename = "../logs/performance_logs.csv";
    CsvWriter csv_writer(log_filename);

    std::vector<std::string> log_columns = {"n_threads", "time"};
    csv_writer.add_line(log_columns);

    for (int n_threads = 1; n_threads < 10; ++n_threads) {
        auto start_time = get_current_time_fenced();
        RandomForest model(100, 0.1, 100, 42, n_threads);
        model.fit(train_dataset);

        auto train_preds = model.predict(train_dataset.X);
        auto train_score = accuracy_score(train_dataset.y, train_preds);

        auto val_preds = model.predict(val_dataset.X);
        auto val_score = accuracy_score(val_dataset.y, val_preds);

        auto finish_time = get_current_time_fenced();
        auto total_time = finish_time - start_time;

        std::cout << "n threads: " << n_threads << \
                  ", total time: " << to_sec(total_time) << std::endl;
        std::cout << "train score: " << train_score << \
          ", validation score: " << val_score << std::endl;

        std::vector<std::string> log_values = {std::to_string(n_threads), std::to_string(to_sec(total_time))};
        csv_writer.add_line(log_values);
    }
    csv_writer.close();

    return 0;
}

