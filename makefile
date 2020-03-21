SYCLFLAGS	:=	-fsycl
MKLFLAGS	:=	-I$(MKLROOT)/include -foffload-static-lib=${MKLROOT}/lib/intel64/libmkl_sycl.a -Wl,-export-dynamic  -Wl,--start-group ${MKLROOT}/lib/intel64/libmkl_intel_ilp64.a ${MKLROOT}/lib/intel64/libmkl_tbb_thread.a ${MKLROOT}/lib/intel64/libmkl_core.a -Wl,--end-group -lsycl -lOpenCL -ltbb -lpthread -lm -ldl

helloSYCL: helloSYCL.cpp
	clang++ -g3 $(SYCLFLAGS) $(MKLFLAGS) $< -o $@

clean:
	rm -f helloSYCL