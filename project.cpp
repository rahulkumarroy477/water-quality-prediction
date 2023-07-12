#include <iostream>
#include <CL/sycl.hpp>
#include <oneapi/dal/io/csv.hpp>
#include <oneapi/dal/table/row_accessor.hpp>
#include <oneapi/dal/table/homogen.hpp>
#include <oneapi/dal/algo/random_forest.hpp>
#include <oneapi/dal/training.hpp>
#include <oneapi/dal/infer.hpp>
#include <oneapi/dal/data/scaling.hpp>
#include <oneapi/dal/data/standard_scaler.hpp>
#include <oneapi/dal/data/accessor.hpp>
#include <oneapi/dal/table/column_accessor.hpp>

namespace dal = oneapi::dal;
namespace sycl = cl::sycl;

int main() {
    // Read CSV data using oneDAL
    dal::csv::data_source data_source("C:\\Users\\91707\\Desktop\\water_quality_prediction\\dataset.csv");

    dal::csv::read_options read_options;
    read_options.set_delimiter(',');
    dal::table data = dal::csv::read(data_source, read_options);

    // Drop columns from the table
    std::vector<std::int64_t> columns_to_drop = {0, 7, 17, 18, 19, 20, 21, 22};
    data = dal::table::select_columns(data, columns_to_drop);

    // Fill missing values with column means
    dal::table_metadata metadata = data.get_metadata();
    auto means = dal::table::mean(data);
    for (std::int64_t i = 0; i < data.get_column_count(); i++) {
        auto column = dal::homogen_table::wrap(means[i], {1, 1});
        metadata.set_feature<float>(i, dal::detail::feature_info{column});
    }
    data.set_metadata(metadata);

    // Split the data into features and target
    dal::table features = dal::table::select_rows(data, {0, data.get_column_count() - 1});
    dal::table target = dal::table::select_rows(data, {data.get_column_count() - 1});

    // Split the data into training and testing sets
    constexpr double test_size = 0.2;
    auto [train_features, test_features, train_target, test_target] =
        dal::train_test_split(features, target, test_size);

    // Scale the features using MinMaxScaler
    dal::data::standard_scaler scaler;
    auto scaler_model = scaler.train(train_features);
    auto train_scaled_features = scaler_model.transform(train_features);
    auto test_scaled_features = scaler_model.transform(test_features);

    // Convert the scaled features and target to Pandas-like DataFrame
    auto train_scaled_df = dal::row_accessor<const float>(train_scaled_features).pull();
    auto test_scaled_df = dal::row_accessor<const float>(test_scaled_features).pull();
    auto train_target_df = dal::row_accessor<const float>(train_target).pull();
    auto test_target_df = dal::row_accessor<const float>(test_target).pull();

    // Perform Random Forest classification
    dal::decision_forest::train::descriptor<float, dal::decision_forest::task::classification,
            dal::decision_forest::method::dense> desc;
    desc.set_tree_count(100);
    desc.set_features_per_node(train_scaled_features.get_column_count() / 3);
    desc.set_min_observations_in_leaf_node(1);

    auto rf_model = dal::train(desc, train_scaled_features, train_target_df);

    // Make predictions on the testing data
    auto test_responses = dal::infer(rf_model, test_scaled_features);
    auto test_responses_df = dal::row_accessor<const float>(test_responses).pull();

    // Calculate the accuracy of the model
    float accuracy = 0.0;
    for (std::size_t i = 0; i < test_target_df.get_row_count(); i++) {
        if (test_target_df[i][0] == test_responses_df[i][0]) {
            accuracy += 1.0;
        }
    }
    accuracy /= test_target_df.get_row_count();

    std::cout << "Accuracy: " << accuracy << std::endl;

    return 0;
}
