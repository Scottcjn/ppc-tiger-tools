#!/bin/bash
# Build AltiVec Quantum LLM on G4 with linker workaround

cd ~/altivec_llm

echo "🔥 Building AltiVec Quantum LLM on PowerPC G4..."
echo ""

# Compile to object file (this works fine)
echo "Step 1: Compiling to object file..."
gcc -c -O2 -mcpu=970 -maltivec -mabi=altivec \
    -Wall -Wno-unused-function -Wno-unused-const-variable \
    simple_quantum_test.c -o quantum_test.o

if [ $? -ne 0 ]; then
    echo "❌ Compilation failed"
    exit 1
fi

echo "✅ Object file created"
echo ""

# Link manually using ld with correct flag format
echo "Step 2: Linking..."
ld -dynamic -arch ppc \
    -o quantum_test \
    /usr/lib/crt1.o \
    quantum_test.o \
    -lSystem \
    -L/usr/lib

if [ $? -ne 0 ]; then
    echo "❌ Linking failed"
    exit 1
fi

echo "✅ Binary created"
echo ""

# Test it
echo "Step 3: Testing..."
./quantum_test

if [ $? -eq 0 ]; then
    echo ""
    echo "🔥🔥🔥 SUCCESS! AltiVec Quantum LLM is working! 🔥🔥🔥"
else
    echo "❌ Test failed"
    exit 1
fi
