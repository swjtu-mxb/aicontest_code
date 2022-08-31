/* 
 * Compression application using static arithmetic coding
 * 
 * Usage: ArithmeticCompress InputFile OutputFile
 * Then use the corresponding "ArithmeticDecompress" application to recreate the original input file.
 * Note that the application uses an alphabet of 257 symbols - 256 symbols for the byte
 * values and 1 symbol for the EOF marker. The compressed file format starts with a list
 * of 256 symbol frequencies, and then followed by the arithmetic-coded data.
 * 
 * Copyright (c) Project Nayuki
 * 
 * https://www.nayuki.io/page/reference-arithmetic-coding
 * https://github.com/nayuki/Reference-arithmetic-coding
 */

#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>
#include "ArithmeticCoder.hpp"
#include "BitIoStream.hpp"
#include "FrequencyTable.hpp"
#include <stdio.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <stdlib.h>
#include <string.h>

using std::uint32_t;


int main(int argc, char *argv[]) {
	// Handle command line arguments
	if (argc != 3) {
		std::cerr << "Usage: " << argv[0] << " InputFile OutputFile" << std::endl;
		return EXIT_FAILURE;
	}
	const char *inputFile  = argv[1];
	const char *outputFile = argv[2];
 
  
  //cout << in.gcount()<<"bytes have been read" << endl;
  //cout <<endl;
    //
  
	// Read input file once to compute symbol frequencies
	std::cout << "start read" << std::endl;
	int fd = open(inputFile, O_RDONLY);
	SimpleFrequencyTable freqs(std::vector<uint32_t>(513, 0));
	freqs.increment(512);  // EOF symbol gets a frequency of 1
 
  using namespace std;
  float arr[64][16][16]={0};
  
  int res = read(fd, (char*) arr, sizeof arr);
	close(fd);

	// for(auto i = 0;i < 16;++i){
	// 	std::cout << arr[0][0][i] << " ";
	// }
	std::cout << std::endl;
  
  using namespace std;
  
  
  for(int i =0;i<64;i++)
  {
    for(int j=0;j<16;j++)
    {
      for(int m=0;m<16;m++){
      
        //b = arr[i][j][m];
        if (arr[i][j][m] == EOF)
			    break;
		    if (arr[i][j][m] < 0 || arr[i][j][m] > 511)
			    throw std::logic_error("Assertion error");
        //cout << arr[i][j][m] << ",";
        //cout << endl;
        freqs.increment(static_cast<uint32_t>(arr[i][j][m]));
      }
    }
  }
  
  
	/*while (true) {
		int b = in.get();
		if (b == EOF)
			break;
		if (b < 0 || b > 511)
			throw std::logic_error("Assertion error");
    cout << b << ",";
    cout << endl;
		freqs.increment(static_cast<uint32_t>(b));
	}*/
	
	// Read input file again, compress with arithmetic coding, and write output file
	fd = open(outputFile, O_WRONLY);

	BitOutputStream bout(fd);
	try {
		
		// Write frequency table
		for (uint32_t i = 0; i < 512; i++) {
			uint32_t freq = freqs.get(i);
			for (int j = 31; j >= 0; j--)
				bout.write_bit(static_cast<int>((freq >> j) & 1));  // Big endian
		}
		
		ArithmeticEncoder enc(32, bout);
   
		/*while (true) {
			// Read and encode one byte
			int symbol = in.get();
			if (symbol == EOF)
				break;
			if (symbol < 0 || symbol > 511)
				throw std::logic_error("Assertion error");
			enc.write(freqs, static_cast<uint32_t>(symbol));
		}*/
   
    for(int i =0;i<64;i++)
    {
      for(int j=0;j<16;j++)
      {
       for(int m=0;m<16;m++){
      
        //b = arr[i][j][m];
        if (arr[i][j][m] == EOF)
			    break;
		    if (arr[i][j][m] < 0 || arr[i][j][m] > 511)
			    throw std::logic_error("Assertion error");
        //cout << arr[i][j][m] << ",";
        //cout << endl;
        enc.write(freqs, static_cast<uint32_t>(arr[i][j][m]));
      }
     }
    }
		
		enc.write(freqs, 512);  // EOF
		enc.finish();  // Flush remaining code bits
		bout.finish();
		close(fd);
		return EXIT_SUCCESS;
		
	} catch (const char *msg) {
		std::cerr << msg << std::endl;
		return EXIT_FAILURE;
	}
}
