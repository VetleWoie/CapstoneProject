To run this project you need to have buildt opencv with the ```NONFREE``` option enabled, and the corrsponding python bindings.

If you don't want to build it locally you can use the docker file in this repo, but keep in mind it takes a while to compile.
On my lenovo thinkcenter it takes roughly an hour and a half to build it.

To build the docker container do:
```docker build --progress=plain --no-cache -t liaci_experiments [PATH TO FOLDER]```

Then to run the experiments do:
```docker run liaci_experiments:latest```

However, for this to work you need a folder called ```image_series``` within this repo, containing a images from a video with names formated as ```image_XXX.png```.
Where ```XXX``` is the frame number of the image.
