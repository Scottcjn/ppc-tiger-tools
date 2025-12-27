# HRM Inference for PowerPC G5 - AltiVec Optimized
# Requires Mac OS X 10.5+ with Accelerate framework and GCC 7+/10

# Note: -O2 and -O3 cause bus errors on G5 due to auto-vectorization
# -O1 is the highest safe optimization level
CXX = g++-10
CXXFLAGS = -O1 -mcpu=970 -mtune=970 -Wall
LDFLAGS = -framework Accelerate

TARGET = hrm_inference
SRC = hrm_inference.cpp

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SRC) $(LDFLAGS)

clean:
	rm -f $(TARGET)

test: $(TARGET)
	./$(TARGET) weights_be

benchmark: $(TARGET)
	@echo "=== C++ AltiVec Benchmark ==="
	./$(TARGET) weights_be
	@echo ""
	@echo "=== Python NumPy Benchmark ==="
	cd ~/trm-sophiacord && PYTHONPATH=/Library/Python/2.5/site-packages python hrm_g5_raw.py

.PHONY: all clean test benchmark
