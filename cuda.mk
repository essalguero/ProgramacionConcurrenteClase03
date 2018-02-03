# vim: set noexpandtab:

# Location of CUDA
CUDA        = /usr/local/cuda
CUDASDK     = /usr/local/cuda/NVIDIA_GPU_Computing_SDK

PATH       += ${CUDA}/open64/bin:${CUDA}/bin
ARCH        = $(shell uname -m)
ifeq (${ARCH},x86_64)
  CUDALIB   = lib64
else
  CUDALIB   = lib
endif

# Variables
CUCC        = nvcc
CUPPFLAGS   = -I${CUDA}/include -I${CUDA}/include/crt -I${CUDASDK}/C/common/inc -I${CUDASDK}/C/common/inc/GL -I../../common/include 
CUFLAGS     =
CULOADLIBES = -L${CUDA}/${CUDALIB} -L${CUDASDK}/C/lib -L${CUDASDK}/C/common/lib/linux -L../../common/lib
CULDLIBS    = -lcuda -lcudart -lcutil_${ARCH} -lglut
CULDFLAGS   =

# Macros for compiling and linking
COMPILE.cu = $(CUCC) $(CUPPFLAGS) $(CUFLAGS) $(TARGET_ARCH)
LINK.cu    = $(CUCC) $(CUPPFLAGS) $(CUFLAGS) $(CULDFLAGS) $(TARGET_ARCH)

# CUDA rules
%.cubin: %.cu
	$(COMPILE.cu) --cubin $<

%.o: %.cu
	$(COMPILE.cu) -c $<

%: %.cu
	$(LINK.cu) $^ $(CULOADLIBES) $(CULDLIBS) -o $@
