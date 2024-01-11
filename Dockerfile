# syntax=docker/dockerfile:1
#FROM python:3.6.8 
FROM ansonxing168/sentencepiece
COPY requirements.txt .
VOLUME /model
#RUN wget https://github.com/google/sentencepiece.git
#RUN apt-get install cmake build-essential pkg-config libgoogle-perftools-dev
#RUN cd sentencepiece
#RUN mkdir build && cd build
#RUN apt-get install cmake
#RUN cmake ..
#RUN make -j $(nproc)
#RUN sudo make install
#RUN sudo ldconfig -v
CMD pip install -r requirements.txt
COPY src/ . 
CMD [ "python", "./mt_server.py", "args.json" ]
