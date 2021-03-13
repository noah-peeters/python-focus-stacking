import numpy as np
import tempfile

# create a memmap array
input = np.memmap(tempfile.NamedTemporaryFile(), dtype='float64', shape=(1000000, 800, 800), mode='w+')
print("ok")
test = [input]
print(np.array(test, dtype="uint8"))
# create a memmap array to store the output
output = np.memmap('output', dtype='float64', shape=(10000,800,800), mode='w+')

def iterate_efficiently(input, output, chunk_size):
    # create an empty array to hold each chunk
    # the size of this array will determine the amount of RAM usage
    holder = np.zeros([chunk_size,800,800], dtype='uint16')

    # iterate through the input, replace with ones, and write to output
    for i in range(input.shape[0]):
        if i % chunk_size == 0:
            holder[:] = input[i:i+chunk_size] # read in chunk from input
            holder += 5 # perform some operation
            output[i:i+chunk_size] = holder # write chunk to output

#iterate_efficiently(input, output, 10)

print("done")