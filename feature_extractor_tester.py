import cv2
import numpy as np
from typing import Type
import os
import time
from tqdm import tqdm

class VideoException(Exception):
    pass

class Feature_extractor_tester():
    def __init__(self, video_path:str, 
                feature_extractor:Type[cv2.Feature2D], 
                descriptor_matcher:Type[cv2.DescriptorMatcher],
                camera_matrix:Type[np.array] = None
                ) -> None:

        self.video_path = video_path
        # if not os.path.exists(video_path):
            # raise FileNotFoundError("Could not find video file")
        self.video_frames = cv2.VideoCapture(self.video_path)
        self.feature_extractor = feature_extractor
        self.descriptor_matcher = descriptor_matcher
        self.camera_matrix = camera_matrix
    
    def find_features(self, frame_1:Type[np.array], frame_2:Type[np.array]) -> tuple[tuple[Type[np.array], Type[np.array]], tuple[Type[np.array], Type[np.array]]]:
        key_points1, descriptors1 = self.feature_extractor.detectAndCompute(frame_1, None)
        key_points2, descriptors2 = self.feature_extractor.detectAndCompute(frame_2, None)

        return ((key_points1, descriptors1), (key_points2, descriptors2))
    
    def find_matches(self, 
                    descriptors1:Type[np.array],
                    key_points1: Type[np.array],
                    descriptors2:Type[np.array],
                    key_points2: Type[np.array],
                    amount: int = None,
                    percentage = 100) -> tuple[Type[np.array], Type[np.array]]:

        # self.descriptor_matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
        # Find calculate scores for each descriptor
        matches = list(self.descriptor_matcher.match(descriptors1, descriptors2, None))
        
        # Sort matches after best match
        matches.sort(key=lambda match: match.distance, reverse=False)

        # Choose top x percent best matches
        if amount is None:
            matches = matches[:int(len(matches)*(percentage/100))]
        else:
            matches = matches[:amount]

        
        # Find mathced keypoints
        matched_key_points = np.array([(np.float32(key_points1[match.queryIdx].pt), np.float32(key_points2[match.trainIdx].pt)) for match in matches])
        return matched_key_points, matches

    def display_video(self,
                    frame_1: Type[np.array],
                    key_points1: Type[np.array],
                    frame_2: Type[np.array],
                    key_points2: Type[np.array],
                    matches: Type[np.array],
                    ):
        img = cv2.drawMatches(frame_1, key_points1, frame_2, key_points2, matches, None)
        cv2.imshow("video", img)
        
        #Try to display image at 30 fps
        wait = cv2.waitKey(1000//60)
        if wait == 27:
            raise KeyboardInterrupt

    def find_trajectory(self, benchmark_file:str = None,  display_video:bool = False, find_translation:bool = True, skip_frame:int = 0):
        #Should we write to file?
        if benchmark_file is not None:
            data_file = open(benchmark_file, "w")
            data_file.write("Time,Features,X,Y,Z\n")
        else:
            data_file = False

        ret, original_frame = self.video_frames.read()
        if not ret:
            raise VideoException("Could not load frame from video")
        
        # print(original_frame)
        # print(ret)
        # exit()
        # For each pair of frames we will calculate the relative translation
        # and orientation from a original frame to a new frame.
        original_frame_gray = cv2.cvtColor(original_frame, cv2.COLOR_BGR2GRAY)

        #Find keypoints in first frame
        keypoint_timer_start = time.perf_counter()
        key_points_original, descriptors_original = self.feature_extractor.detectAndCompute(original_frame_gray, None)
        keypoint_timer_end = time.perf_counter()
        
        #Write benchmarkdata to file
        if data_file:
            data_file.write(f"{keypoint_timer_end-keypoint_timer_start},{len(key_points_original)},-,-,-\n")

        #Loop for as long as there are frames in the video
        with tqdm(total=(int(self.video_frames.get(cv2.CAP_PROP_FRAME_COUNT))-1)//skip_frame) as pbar:
            while ret:
                ret, new_frame = self.video_frames.read()
                for i in range(skip_frame):
                    self.video_frames.read()
                if not ret:
                    break
                new_frame_gray = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)

                #Find keypoints in second frame
                keypoint_timer_start = time.perf_counter()
                key_points_new, descriptors_new = self.feature_extractor.detectAndCompute(new_frame_gray, None)
                keypoint_timer_end = time.perf_counter()

                #If there are not enough keypoint in either image skip the image pairs
                t = np.array([["-"],["-"],["-"]])
                if not (len(key_points_new) <= 5):
                    #Find matches between keypoints
                    matched_key_points, matches = self.find_matches(descriptors_original, key_points_original, descriptors_new, key_points_new, percentage=40)

                    if display_video:
                        self.display_video(original_frame, key_points_original, new_frame, key_points_new, matches)

                    #Get relative translation and rotation between the frames
                    if find_translation:
                        if len(matches) > 5:
                            essential, mask = cv2.findEssentialMat(matched_key_points[:,0,:], matched_key_points[:,1,:], self.camera_matrix)
                            R1, R2, t = cv2.decomposeEssentialMat(essential)            
                        
                    # Copy new frame into old frame
                    original_frame = new_frame.copy()
                    original_frame_gray = new_frame_gray.copy()
                    key_points_original = key_points_new
                    descriptors_original = descriptors_new

                if data_file:
                        data_file.write(f"{keypoint_timer_end-keypoint_timer_start},{len(key_points_new)},{t[0,0]},{t[1,0]},{t[2,0]}\n")

                pbar.update(1)

        if data_file:
            data_file.close()

if __name__=="__main__":

    # projection_parameters:
    fx = 8.024483249061652259e+02
    fy = 6.356353010700777304e+02
    cx = 8.032274574101340932e+02
    cy = 3.974427528375753695e+02
    cameraMatrix = np.array([
                        [fx, 0, cx],
                        [0, fy, cy], 
                        [0,  0,  1]
                    ])
    result_path = os.environ["EXPERIMENTAL_RESULTS"]
    # video_path = os.environ["DATADIR"] + "/the_wall.mp4"#../data/the_wall/thewall.mp4"
    video_path = os.environ["DATADIR"] + "/image_series/image_%d.png"#../data/the_wall/thewall.mp4"


    # print()
    # print("SURF benchmark: No Cap")
    # sift_bench = Feature_extractor_tester(video_path=video_path, feature_extractor=cv2.xfeatures2d.SURF_create(5000), descriptor_matcher=cv2.BFMatcher(), camera_matrix = cameraMatrix)
    # sift_bench.find_trajectory(display_video=False, find_translation=True, benchmark_file = f"{result_path}/surf_bench_nocap.csv")

    # print("ORB benchmark: No Cap")
    # orb_bench = Feature_extractor_tester(video_path=video_path, feature_extractor=cv2.ORB_create(5000), descriptor_matcher=cv2.BFMatcher(), camera_matrix = cameraMatrix)
    # orb_bench.find_trajectory(display_video=False, find_translation=True, benchmark_file = f"{result_path}/orb_bench_nocap.csv")

    # print("SIFT benchmark: No Cap")
    # sift_bench = Feature_extractor_tester(video_path=video_path, feature_extractor=cv2.SIFT_create(5000), descriptor_matcher=cv2.BFMatcher(), camera_matrix = cameraMatrix)
    # sift_bench.find_trajectory(display_video=False, find_translation=True, benchmark_file = f"{result_path}/sift_bench_nocap.csv")

    for cap in [300,400,500]:
        for features_for_estimation in [10,30,40,100]:
            print(f"SURF benchmark: Cap = {cap} Features_For_Estimation = {features_for_estimation}")
            sift_bench = Feature_extractor_tester(video_path=video_path, feature_extractor=cv2.xfeatures2d.SURF_create(cap), descriptor_matcher=cv2.BFMatcher(), camera_matrix = cameraMatrix)
            sift_bench.find_trajectory(display_video=False, find_translation=True, benchmark_file = f"{result_path}/surf_bench_cap={cap}_featuresForEstimation={features_for_estimation}.csv", skip_frame=25)

            print(f"ORB benchmark: Cap = {cap} Features_For_Estimation = {features_for_estimation}")
            orb_bench = Feature_extractor_tester(video_path=video_path, feature_extractor=cv2.ORB_create(cap), descriptor_matcher=cv2.BFMatcher(), camera_matrix = cameraMatrix)
            orb_bench.find_trajectory(display_video=False, find_translation=True, benchmark_file = f"{result_path}/orb_bench_cap={cap}_featuresForEstimation={features_for_estimation}.csv", skip_frame=25)

            print(f"SIFT benchmark: Cap = {cap} Features_For_Estimation = {features_for_estimation}")
            sift_bench = Feature_extractor_tester(video_path=video_path, feature_extractor=cv2.SIFT_create(cap), descriptor_matcher=cv2.BFMatcher(), camera_matrix = cameraMatrix)
            sift_bench.find_trajectory(display_video=False, find_translation=True, benchmark_file = f"{result_path}/sift_bench_cap={cap}_featuresForEstimation={features_for_estimation}.csv", skip_frame=25)
    