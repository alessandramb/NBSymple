SOURCES = main.cu particle.cu   
OBJS = main.o particle.o 

NVCC = nvcc
EXE = NBSymple
NVCCFLAGS = -D_OMP -Xcompiler "-fopenmp" -Xlinker "-lgomp"

ifdef DEBUG
  NVCCFLAGS += -g -G
endif

ifdef L 
  NVCCFLAGS += -DLEAPFROG_METHOD
  SOURCES   += leapfrog.cu 
  OBJS      += leapfrog.o   
endif

ifdef S 
  NVCCFLAGS += -DSIXTH_METHOD -arch sm_13
  SOURCES   += sixth.cu 
  OBJS      += sixth.o 
endif

ifdef DS
  NVCCFLAGS += -DDS_SIXTH
  SOURCES   += sixth.cu
  OBJS      += sixth.o
endif

target: all
	NAME_DIR=Data ; \
        if ! [ -d $$NAME_DIR ]; then \
                mkdir -p $$NAME_DIR; \
        fi


all : $(OBJS)
	$(NVCC) -o $(EXE) $(NVCCFLAGS) $(OBJS)

%.o : %.cu
	$(NVCC) -c $^ $(NVCCFLAGS) -o $@

clean:
	rm *.o $(EXE)
