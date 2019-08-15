LIB_DIR = FiniteStateEntropy/lib

default: pyfse

pyfse: setup.py pyfse.pyx $(LIB_DIR)/libfse.a
	python3 setup.py build_ext --inplace && rm -f pyfse.c && rm -Rf build

$(LIB_DIR)/libfse.a:
	make -C $(LIB_DIR) libfse.a

clean:
	rm *.so pyfse.c
