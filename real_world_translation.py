import pandas as pd
import numpy as np

def compute_translation(datafile):
    data = pd.read_csv(datafile)
    x = data["x"].to_numpy()
    y = data["y"].to_numpy()
    z = data["z"].to_numpy()
    
    positions = np.array([x,y,z]).T

    og = positions[0]
    with open("real_world_translations_skipframe=0.csv","w") as file:
        file.write("x,y,z\n,,\n")
        for pos in positions[1:]:
            translation = (pos-og)/np.linalg.norm(pos-og)
            file.write(f"{translation[0]},{translation[1]},{translation[2]}\n")
            og = pos.copy()

    

if __name__ == "__main__":
    compute_translation("../data/the_wall/position_the_wall.csv")
