#!/bin/bash
# Add I2_S support to ggml.c
# Run from llama.cpp-b2000 directory

# First, add include for I2_S patch header after ggml-impl.h include
grep -q 'ggml-i2s-patch.h' ggml.c || \
sed -i.bak '/#include "ggml-impl.h"/a\
#include "ggml-i2s-patch.h"
' ggml.c

# Add I2_S entry to type_traits after Q8_K entry
# Check if already added
if ! grep -q 'GGML_TYPE_I2_S' ggml.c; then
    # Find the line with Q8_K closing brace and add I2_S after it
    sed -i '/.type_name.*=.*"q8_K"/,/^    }/ {
        /^    }/ a\
    [GGML_TYPE_I2_S] = {\
        .type_name                = "i2_s",\
        .blck_size                = 256,\
        .type_size                = 64,\
        .is_quantized             = true,\
        .to_float                 = NULL,\
        .from_float               = NULL,\
        .from_float_reference     = NULL,\
        .vec_dot                  = NULL,\
        .vec_dot_type             = GGML_TYPE_Q8_K,\
    },
    }' ggml.c
    echo "Added I2_S entry to type_traits"
else
    echo "I2_S entry already exists"
fi

echo "Done. Check ggml.c for I2_S support."
