import src.framework__data_set as ds

print("##### Extracting CPU Data #####")
dataset_cpu = ds.get_data_set(
    metric="container_cpu",
    application_name="collector",
    path_to_data="./data/"
)

dataset_cpu.plot_dataset(number_of_samples=5)



print("##### Extracting Memory Data #####")
import src.framework__data_set as ds
dataset_mem = ds.get_data_set(
    metric="container_mem",
    application_name="collector",
    path_to_data="./data/"
)

for df in dataset_mem:
    import matplotlib.pyplot as plt

    title = df.iloc[0, 2:5].str.cat(sep=', ')
    # plt.close("all")
    ts = df["sample"].copy()
    ts.index = [time for time in df["time"]]
    ts.plot()
    start_date = df["time"].iloc[0]
    end_date = df["time"].iloc[len(df)-1]
    plt.title(title + "\n From: " + str(start_date)+"  To: " + str(end_date))
    plt.show()


dataset_mem.plot_dataset(number_of_samples=30)
