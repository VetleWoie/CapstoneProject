FROM ubuntu:22.04

ENV DATADIR=/app/data
ENV EXPERIMENTAL_RESULTS=/app/results



RUN apt-get update && \
    apt-get install -y cmake gcc g++ git \
    python3-dev python3-pip \
    libavcodec-dev libavformat-dev libswscale-dev \
    libgstreamer-plugins-base1.0-dev libgstreamer1.0-dev libgtk-3-dev 

# https://stackoverflow.com/questions/70334087/how-to-build-opencv-from-source-with-python-binding
# RUN apt-get update && apt-get install git \
#     build-essential pkg-config libboost-system-dev libboost-thread-dev \
#     libboost-program-options-dev libboost-test-dev \
#     libgl1-mesa-glx libglib2.0-0 \
#     libjpeg-dev libpng-dev libtiff-dev \
#     libavcodec-dev libavformat-dev libswscale-dev  \
#     libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
#     libxvidcore-dev x264 libx264-dev libfaac-dev libmp3lame-dev libtheora-dev \
#     libfaac-dev libmp3lame-dev libvorbis-dev \
#     libtbb-dev libatlas-base-dev -y

RUN git clone https://github.com/opencv/opencv.git
RUN cd /tmp && git clone https://github.com/opencv/opencv_contrib.git

RUN pip3 install numpy

RUN cd /opencv && mkdir build && cd build &&\
    cmake \
        -D CMAKE_BUILD_TYPE=RELEASE \
        -D CMAKE_INSTALL_PREFIX=$(python3 -c "import sys; print(sys.prefix)") \
        -D INSTALL_PYTHON_EXAMPLES=OFF \
        -D INSTALL_C_EXAMPLES=OFF \
        -D WITH_TBB=ON \
        -D WITH_CUDA=OFF \
        -D BUILD_opencv_cudacodec=OFF \
        -D ENABLE_FAST_MATH=1 \
        -D CUDA_FAST_MATH=1 \
        -D WITH_CUBLAS=1 \
        -D WITH_V4L=ON \
        -D WITH_QT=OFF \
        -D WITH_OPENGL=ON \
        -D WITH_GSTREAMER=ON \
        -D OPENCV_GENERATE_PKGCONFIG=ON \
        -D OPENCV_PC_FILE_NAME=opencv.pc \
        -D OPENCV_ENABLE_NONFREE=ON \
        -D OPENCV_PYTHON3_INSTALL_PATH=$(python3 -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())") \
        -D OPENCV_EXTRA_MODULES_PATH=/tmp/opencv_contrib/modules \
        -D PYTHON_EXECUTABLE=$(which python3) \
        -D BUILD_EXAMPLES=OFF .. 

RUN cd /opencv/build && make
RUN cd /opencv/build && make install

WORKDIR /app
RUN mkdir ${DATADIR}
RUN mkdir ${EXPERIMENTAL_RESULTS}

RUN pip3 install tqdm

COPY ./thewall.mp4 ${DATADIR}/the_wall.mp4
COPY ./image_series ${DATADIR}/image_series
COPY ./feature_extractor_tester.py app.py

CMD ["python3", "app.py"]