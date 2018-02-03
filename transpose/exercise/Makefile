include ../../cuda.mk

transpose: transpose.o transpose_gold.o
	$(LINK.cu) $^ $(CULOADLIBES) $(CULDLIBS) -o $@

clean:
	$(RM) transpose.o transpose_gold.o transpose
