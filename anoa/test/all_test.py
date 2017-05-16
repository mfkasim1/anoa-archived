import anoa.test.basic_test as basic_test
import anoa.test.math_test as math_test
import anoa.test.array_test as array_test
import anoa.test.fftpack_test as fftpack_test

def main():
    basic_test.main()
    basic_test.main(with_shape=False)
    math_test.main()
    array_test.main()
    array_test.main(with_shape=False)
    fftpack_test.main()
    
    print("All tests completed")

if __name__ == "__main__":
    main()
