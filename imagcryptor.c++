#include <iostream>
#include <fstream>
#include <vector>

// Using BMP file header structure as its simple and widely understood
struct BMPHeader {
    char signature[2];
    uint32_t fileSize;
    uint32_t reserved;
    uint32_t dataOffset;
    uint32_t headerSize;
    int32_t width;
    int32_t height;
    uint16_t planes;
    uint16_t bitsPerPixel;
    uint32_t compression;
    uint32_t imageSize;
    int32_t xPixelsPerMeter;
    int32_t yPixelsPerMeter;
    uint32_t colorsUsed;
    uint32_t importantColors;
};

class ImageReader {
public:
    std::vector<uint8_t> pixels;
    int width, height, channels;
    
    bool readBMP(const std::string& filename) {
        std::ifstream file(filename, std::ios::binary);
        if (!file.is_open()) return false;
        
        BMPHeader header;
        file.read(reinterpret_cast<char*>(&header), sizeof(header));
        
        if (header.signature[0] != 'B' || header.signature[1] != 'M') {
            return false; // Not a BMP file
        }
        
        width = header.width;
        height = header.height;
        channels = header.bitsPerPixel / 8;
        
        file.seekg(header.dataOffset);
        pixels.resize(width * height * channels);
        file.read(reinterpret_cast<char*>(pixels.data()), pixels.size());
        
        return true;
    }
};