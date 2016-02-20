CXXFLAGS=\
	-Ileveldb/include \
	-Irocksdb/include \
	-pthread \
	-g -Wall -Wextra -Wsign-conversion -Winline -Wno-unused-function \
	-Wconversion \
	-O3 \
	-march=native \
	-std=c++0x
	# -std=c++11
	# -fno-omit-frame-pointer

MAIN_SRC=util.cpp leveldb.cpp leveldb_impl.cpp rocksdb_impl.cpp meshdb.cpp main.cpp
MEASURE_RW_SRC=measure_rw.cpp

TARGETS=main measure_rw

MAIN_OBJ=$(patsubst %.cpp,%.o,$(MAIN_SRC))
MAIN_DEPFILES:=$(patsubst %.cpp,%.d,$(MAIN_SRC))

MEASURE_RW_OBJ=$(patsubst %.cpp,%.o,$(MEASURE_RW_SRC))
MEASURE_RW_DEPFILES:=$(patsubst %.cpp,%.d,$(MEASURE_RW_SRC))

all: $(TARGETS)

main: $(MAIN_OBJ) leveldb/libleveldb.a rocksdb/librocksdb.a
	$(CXX) $(CXXFLAGS) -o $@ $^ -lsnappy -lz -lbz2 -lrt

measure_rw: $(MEASURE_RW_OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $^

clean:
	$(RM) $(MAIN_OBJ) $(MAIN_DEPFILES) $(MEASURE_RW_OBJ) $(MEASURE_RW_DEPFILES) $(TARGETS)


# dependancy checking from https://stackoverflow.com/a/313787
NODEPS:=clean

ifeq (0, $(words $(findstring $(MAKECMDGOALS), $(NODEPS))))
	-include $(DEPFILES)
endif

%.d: %.cpp
	$(CXX) $(CXXFLAGS) -MM -MT '$(patsubst %.cpp,%.o,$<)' $< -MF $@
# end


