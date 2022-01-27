from project.src.project_utils import set_seed_for_random
from project.src.model import Model
from project.src.grouped_data import GroupedData


if __name__ == "__main__":
    processed_data_path = './data/processed/'
    set_seed_for_random(6)

    # =========== Model with true data from santa. (Initial loading can take a minute)
    model = Model(8, 1000, _selection_size=4, _pheromone_weight=2, _distance_weight=4, _size_weight=2, _home_weight=1)
    model.load_data(GroupedData(processed_data_path + 'points.csv', processed_data_path + 'groups.csv'))
    routes, score = model.search_routes(120, gen_counter=5)

    # =========== Model with synthetic data. Good for testing how this model manage to do it's job.
    # model = Model(8, 100, _selection_size = 2, _pheromone_weight=2, _distance_weight=4, _size_weight=3, _home_weight=1)
    # model.load_data(GroupedData(n_points=100, box_size=20, group_size= 4))
    # routes, score = model.search_routes(50, gen_counter=3)
    # model.show_routes(routes[:5])

    # ============== Show results
    print(f"Routes: {routes}")
    print(f"Score: {score}")
