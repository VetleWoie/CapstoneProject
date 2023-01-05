from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

def plot_time_per_frame(datafiles, legends, datafolder = ""):
    fig, ax = plt.subplots(1,1)
    
    for file, legend in zip(datafiles, legends):
        data = pd.read_csv(datafolder +"/"+ file)
        ax.plot(data["Time"], label=legend)
        ax.set_xlabel("Frame number")
        ax.set_ylabel("Time/$\mu s$")
        ax.legend(loc='upper right')
    plt.savefig("time_data.pdf")
    plt.show()

def plot_features_per_frame(datafiles, legends, datafolder = ""):
    fig, ax = plt.subplots(1,1)
    
    for file ,legend in zip(datafiles,legends):
        data = pd.read_csv(datafolder +"/"+ file)
        ax.plot(data["Features"], label=legend)
        ax.set_xlabel("Frame number")
        ax.set_ylabel("Number of features")
        ax.legend(loc='upper right')
    plt.savefig("feature_data.pdf")
    plt.show()

def plot_time_per_feature_per_frame(datafiles, legends, datafolder = ""):
    fig, ax = plt.subplots(1,1)
    
    for file, legend in zip(datafiles,legends):
        data = pd.read_csv(datafolder +"/"+ file)
        ax.plot(data["Time"]/data["Features"], label=legend)
        ax.set_xlabel("Frame number")
        ax.set_ylabel("Time/Features")
        ax.legend(loc='upper right')
    plt.savefig("time_per_featrue_data.pdf")
    plt.show()

def plot_error_distance_per_frame(datafiles, legends,skipped_frames = 1, realdata = None, datafolder = ""):
    true_data = pd.read_csv(realdata, na_values=['-', '']).to_numpy()
    fig, ax = plt.subplots(1,1)

    for file, legend in zip(datafiles,legends):
        data = pd.read_csv(datafolder+"/"+file, na_values=['-', '']).to_numpy()[:,2:]
        data = data[:,[0,1,2]]
        error_distance = (true_data-data)
        error_distance = np.sqrt((error_distance*error_distance).sum(axis=1))
        
        ax.plot(np.arange(len(error_distance))*skipped_frames,error_distance, label = legend)
        ax.set_xlabel("Frame Number")
        ax.set_ylabel("Error distance")
        ax.legend()
    plt.savefig("error_data.pdf")
    plt.show()
    
    

if __name__=="__main__":
    datafolder = "./experimental_data/all_data"

    plot_time_per_frame([
                            "/orb_bench_nocap.csv",
                            "/sift_bench_nocap.csv",
                            "/surf_bench_nocap.csv",
                            "/surf_bench_cap=300_featuresForEstimation=10.csv",
                            "/surf_bench_cap=400_featuresForEstimation=10.csv",
                            "/surf_bench_cap=500_featuresForEstimation=10.csv",
                        ],
                        legends = ["ORB", "SIFT", "SURF t=0","SURF t=300", "SURF t=400","SURF t=500"], 
                        datafolder=datafolder
                        )
    plot_features_per_frame([
                            "/orb_bench_nocap.csv",
                            "/sift_bench_nocap.csv",
                            "/surf_bench_nocap.csv",
                            "/surf_bench_cap=300_featuresForEstimation=10.csv",
                            "/surf_bench_cap=400_featuresForEstimation=10.csv",
                            "/surf_bench_cap=500_featuresForEstimation=10.csv",
                        ],
                        legends = ["ORB", "SIFT", "SURF t=0","SURF t=300", "SURF t=400","SURF t=500"], 
                        datafolder=datafolder
                        )
    plot_time_per_feature_per_frame([
                            "/orb_bench_nocap.csv",
                            "/sift_bench_nocap.csv",
                            "/surf_bench_nocap.csv",
                            "/surf_bench_cap=300_featuresForEstimation=10.csv",
                            "/surf_bench_cap=400_featuresForEstimation=10.csv",
                            "/surf_bench_cap=500_featuresForEstimation=10.csv",
                        ],
                        legends = ["ORB", "SIFT", "SURF t=0","SURF t=300", "SURF t=400","SURF t=500"], 
                        datafolder=datafolder
                        )

    real_world_data =  "./real_world_translations_skipframe=25.csv"

    plot_error_distance_per_frame(
                                [
                                    "orb_bench_cap=500_featuresForEstimation=10.csv",
                                    "sift_bench_cap=500_featuresForEstimation=10.csv",
                                    "surf_bench_nocap.csv",
                                    "surf_bench_cap=300_featuresForEstimation=10.csv",
                                    "surf_bench_cap=400_featuresForEstimation=10.csv",
                                    "surf_bench_cap=500_featuresForEstimation=10.csv",
                                ],
                                ["ORB", "SIFT", "SURF t=300", "SURF t=400","SURF t=500"],
                                skipped_frames=25,
                                realdata=real_world_data,
                                datafolder=datafolder)