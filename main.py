import yaml
from modules.problem_definition import define_problem
from modules.data_collection import collect_data
from modules.data_preparation import prepare_data
from modules.data_analysis import analyze_data
from modules.model_building import build_model

# Load config
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

def main():
    # 1. Define the problem
    define_problem(config["problem_definition"])

    # 2. Collect data
    data = collect_data(config["data_collection"])

    # 3. Prepare data
    prepared_data = prepare_data(data, config["data_preparation"])

    # 4. Analyze data
    analyze_data(prepared_data, config["data_analysis"])

    # 5. Build and evaluate model
    build_model(prepared_data, config["model_building"])

if __name__ == "__main__":
    main()
