import anoa.test.basic_test as basic_test
import anoa.test.math_test as math_test
import anoa.test.array_test as array_test
import anoa.test.fftpack_test as fftpack_test
import time, cProfile

def main():
    t0 = time.time()
    basic_test.main()
    basic_test.main(with_shape=False)
    math_test.main()
    array_test.main()
    array_test.main(with_shape=False)
    fftpack_test.main()
    t1 = time.time()
    
    print("All tests completed in %f s" % (t1 - t0))

if __name__ == "__main__":
    main()
