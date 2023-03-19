import src.framework__data_set as ds

print("##### Extracting CPU Data #####")
dataset_cpu = ds.get_data_set(
    metric="container_cpu",
    application_name="collector",
    path_to_data="./data/"
)

dataset_cpu.plot_dataset(number_of_samples=5)

print("##### Extracting Memory Data #####")
dataset_mem = ds.get_data_set(
    metric="container_mem",
    application_name="collector",
    path_to_data="./data/"
)

dataset_mem.plot_dataset(number_of_samples=5)

