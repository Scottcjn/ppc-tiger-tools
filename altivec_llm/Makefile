# Makefile for PowerPC G4/G5 AltiVec Quantum LLM Agent
# Fixed for Mac OS X Leopard

CC = gcc
CFLAGS = -O3 -mcpu=970 -maltivec -mabi=altivec -funroll-loops -ffast-math -Wall -Wextra -Wno-unused-parameter
LDFLAGS = -lcurl -lpthread -lm

all: altivec_llm_agent

altivec_llm_agent: ggml_altivec_quantum.c vec_perm_quantum_patterns.h
	@echo "========================================="
	@echo "Building AltiVec Quantum LLM Agent"
	@echo "========================================="
	@echo "Compiler: $(CC)"
	@echo "CPU Target: 970"
	@echo "Flags: $(CFLAGS)"
	@echo ""
	$(CC) $(CFLAGS) ggml_altivec_quantum.c -o altivec_llm_agent $(LDFLAGS)
	@echo ""
	@echo "✅ Build complete: altivec_llm_agent"
	@./altivec_llm_agent --version || true

clean:
	rm -f altivec_llm_agent *.o

install: altivec_llm_agent
	mkdir -p ~/bin
	cp altivec_llm_agent ~/bin/
	@echo "✅ Installed to ~/bin/altivec_llm_agent"

.PHONY: all clean install
