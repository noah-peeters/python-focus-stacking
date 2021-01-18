import numpy

a = numpy.memmap("test.mymemmap", dtype="float32", mode="w+", shape=(1000, 1000))

del a   # Close file
b = numpy.memmap("tes.mymemmap", dtype="float32", mode="r+", shape=(1000, 1000))