#include "main.h"

#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include <opencv2/imgproc/imgproc.hpp> // cv::cvtColor

#include "timer.h"

#include <array>
#include <iostream>
#include <filesystem>
#include <vector>
#include <string>
#include <CL/sycl.hpp>

// TODO: Figure out why hash for DHash and AHash is negative.

#define LOG(x) { std::cout << x << std::endl; }

using Hash = std::uint64_t;
using Path = std::filesystem::path;

enum class FileType {
    UNKNOWN = -1,
    PICTURE = 0,
    VIDEO = 1,
    HEIC = 2
};

FileType GetType(const Path& extension) {
    if (extension.compare(".jpg") == 0 || extension.compare(".JPG") == 0) {
        return FileType::PICTURE;
    } else if (extension.compare(".mov") == 0 || extension.compare(".MOV") == 0) {
        return FileType::VIDEO;
    } else if (extension.compare(".heic") == 0 || extension.compare(".HEIC") == 0) {
        return FileType::HEIC;
    } else if (extension.compare(".png") == 0 || extension.compare(".PNG") == 0 ||
               extension.compare(".jpeg") == 0 || extension.compare(".JPEG") == 0) {
        return FileType::PICTURE;
    } else if (extension.compare(".mp4") == 0 || extension.compare(".MP4") == 0 ||
               extension.compare(".wmv") == 0 || extension.compare(".WMV") == 0 ||
               extension.compare(".gif") == 0 || extension.compare(".GIF") == 0 ||
               extension.compare(".avi") == 0 || extension.compare(".AVI") == 0) {
        return FileType::VIDEO;
    } else if (extension.compare(".bmp") == 0 || extension.compare(".BMP") == 0) {
        return FileType::PICTURE;
    } else if (extension.compare(".heif") == 0 || extension.compare(".HEIF") == 0) {
        return FileType::HEIC;
    } else {
        return FileType::UNKNOWN;
    }
}

struct File {
    File(const cv::Mat& image, const Path& path, bool flag) : image{ image }, original_image{ image }, path{ path }, flag{ flag } {}
    cv::Mat image;
    cv::Mat original_image;
    Path path;
    bool flag;
};

Hash GetAHash(const cv::Mat& resized_image) {
    Hash hash{ 0 };
    int i{ 0 }; // counts the current index in the hash
    // Cycle through every row
    std::size_t sum{ 0 };
    std::size_t pixels{ static_cast<std::size_t>(resized_image.rows * resized_image.cols) };
    // Find average of all pixel brightnesses
    for (int index = 0; index < pixels; ++index) {
        // Get pixel pointer for the row
        auto ptr = (std::uint8_t*)(resized_image.data + index);
        sum += *ptr;
    }
    std::size_t average{ sum / pixels };
    for (int y = 0; y < resized_image.rows; ++y) {
        // Get pixel pointer for the row
        auto ptr = (std::uint8_t*)(resized_image.data + y * resized_image.step);
        // Cycle through every column
        for (int x = 0; x < resized_image.cols; ++x) {
            if (ptr[x] > average) {
                hash |= (Hash)1 << i;
            }
            i++; // increment hash index
        }
    }
    assert(hash != 0);
    return hash;
}

Hash GetDHash(const cv::Mat& resized_image) {
    assert(resized_image.rows == resized_image.cols - 1);
    Hash hash{ 0 };
    int i{ 0 }; // counts the current index in the hash
    // Cycle through every row 
    for (int y = 0; y < resized_image.rows; ++y) {
        // Get pixel pointer for the row
        auto ptr = (std::uint8_t*)(resized_image.data + y * resized_image.step);
        // Cycle through every column
        for (int x = 0; x < resized_image.cols - 1; ++x) {
            // If the next pixel is brighter, make the hash contain a 1, else keep it as 0
            if (ptr[x + 1] > ptr[x]) {
                hash |= (Hash)1 << i;
            }
            i++; // increment hash index
        }
    }
    assert(hash != 0);
    return hash;
}

inline std::uint64_t GetLinearIndex(std::size_t hash_count, std::uint64_t i, std::uint64_t j) {
    return hash_count * (hash_count - 1) / 2 - (hash_count - j) * ((hash_count - j) - 1) / 2 + i - j - 1;
}

cv::Mat Resize(const cv::Mat& image, int width, int height) {
    cv::Mat resized;
    cv::resize(image, resized, { width, height });
    return resized;
}

void AddBorder(cv::Mat& image, int size, const cv::Scalar& color) {
    cv::Rect inner_rectangle{ size, size, image.cols - size * 2, image.rows - size * 2 };
    cv::Mat border;
    border.create(image.rows, image.cols, image.type());
    border.setTo(color);
    image(inner_rectangle).copyTo(border(inner_rectangle));
    image = border;
}

cv::Mat ToGreyscale(const cv::Mat& image) {
    cv::Mat greyscale;
    cv::cvtColor(image, greyscale, cv::ColorConversionCodes::COLOR_BGR2GRAY);
    return greyscale;
}

inline cv::Mat GetGreyImage(const Path& path) {
    switch (GetType(path.extension())) {
        case FileType::PICTURE:
        {
            return cv::imread(path.generic_string(), cv::IMREAD_GRAYSCALE);
        }
        case FileType::VIDEO:
        {

            return {};
        }
        case FileType::HEIC:
        {

            return {};
        }
        case FileType::UNKNOWN:
            return {};
    }
    return {};
}

cv::Mat GetImage(const Path& path) {
    switch (GetType(path.extension())) {
        case FileType::PICTURE:
        {
            return cv::imread(path.generic_string());
        }
        case FileType::VIDEO:
        {

            return {};
        }
        case FileType::HEIC:
        {

            return {};
        }
        case FileType::UNKNOWN:
            return {};
    }
    return {};
}

std::vector<Hash> GetHashes(const std::vector<Path>& files) {
    std::size_t file_count{ files.size() };
    
    LOG("[Starting hashing of " << file_count << " files]");

    std::vector<Hash> hashes;
    hashes.resize(file_count, 0);

    std::size_t percent{ file_count / 100 };
    
    for (auto i{ 0 }; i < file_count; ++i) {
        const auto& path{ files[i] };
        auto image{ GetGreyImage(path) };
        if (!image.empty()) {
            auto thumbnail{ Resize(image, 8, 8) };
            hashes[i] = GetAHash(thumbnail);
        }
        if (i % percent == 0) {
            LOG("[" << i / percent << "% of files hashed]");
        }
    }
    
    LOG("[Finished hashing all files]");
    return hashes;
}

std::vector<Path> GetFiles(const std::vector<const char*>& directories) {
    LOG("[Starting file search through " << directories.size() << " directories]");
    std::vector<Path> files;
    for (const auto& directory : directories) {
        // Cycle through each file in the given directory.
        for (const auto& file : std::filesystem::directory_iterator(directory)) {
            if (file.is_regular_file()) {
                files.emplace_back(std::filesystem::absolute(file.path()));
            }
        }
    }
    LOG("[Finished file search]");
    return files;
}

inline std::uint64_t GetHammingDistance(Hash hash1, Hash hash2) {
    return sycl::popcount(hash1 ^ hash2); // XOR (get number of 1 bits in the difference of 1 and 2).
}

using Pair = std::uint8_t;
using PairContainer = std::vector<Pair>;

PairContainer ProcessDuplicates(std::vector<Hash>& hashes, int hamming_threshold) {
    std::size_t hash_count{ hashes.size() };
    LOG("[Starting duplicate search through " << hash_count << " hashes]");

    std::size_t pair_count{ hash_count * (hash_count - 1) / 2 };
    
    PairContainer pairs;
    pairs.resize(pair_count, 0);
    
    cl::sycl::queue queue;
    cl::sycl::range<1> destination_range{ pair_count };
    cl::sycl::range<1> hash_range{ hash_count };
    cl::sycl::property_list properties{ cl::sycl::property::buffer::use_host_ptr {} };
    
    {
        cl::sycl::buffer<Hash, 1> hash_b{ hashes, properties };
        cl::sycl::buffer<Pair, 1> destination_b{ pairs, properties };

        queue.submit([&](cl::sycl::handler& cgh) {
            auto destination_a = destination_b.template get_access<cl::sycl::access::mode::write>(cgh);
            auto hash_a = hash_b.template get_access<cl::sycl::access::mode::read>(cgh);

            auto kernel = [=](cl::sycl::id<1> id) {
                auto i{ id.get(0) };
                for (auto j{ 0 }; j < i; ++j) {
                    auto index = GetLinearIndex(hash_count, i, j);
                    auto hamming_distance{ GetHammingDistance(hash_a[i], hash_a[j]) };
                    if (hamming_distance <= hamming_threshold) {
                        //int similarity{ static_cast<int>(1.0 - static_cast<double>(hamming_distance) / (static_cast<double>(hamming_threshold) + 1.0) * 100.0) };
                        destination_a[index] = hamming_distance + 1;
                    }
                }
            };

            cgh.parallel_for<class find_similar_images>(hash_range, kernel);
        });
        queue.wait();
    }
    LOG("[Finished duplicate search]");
    return pairs;
}

int main(int argc, char** argv) {

    //auto files{ GetFiles({ "../vizsla_154/"}) };//, "../maltese_252/", "../vizsla_4048/" }) };
    //auto files{ GetFiles({ "../test/" }) };
    auto files{ GetFiles({ "D:/Media/All" }) };
    auto hashes{ GetHashes(files) };
    auto pairs{ ProcessDuplicates(hashes, 0) };

    cv::Size window{ 800, 400 };
    auto hash_count{ hashes.size() };

    LOG("[Displaying duplicate pairs]");

    for (auto i{ 0 }; i < hash_count; ++i) {
        for (auto j{ 0 }; j < i; ++j) {
            auto index = GetLinearIndex(hash_count, i, j);
            auto hamming_distance = pairs[index] - 1;
            if (hamming_distance >= 0) {
                assert(i < files.size());
                assert(j < files.size());
                auto image1 = GetImage(files[i]);
                auto image2 = GetImage(files[i]);
                if (!image1.empty() && !image2.empty()) {
                    LOG("[" << (int)(index / pairs.size() * 100) << "% of files displayed]");
                    std::array<File, 2> images{
                        File(Resize(image1, window.width / 2, window.height), files[i], false),
                        File(Resize(image2, window.width / 2, window.height), files[j], true),
                    };

                    AddBorder(images[0].image, 10, { 0, 255, 0 });
                
                    cv::Mat concatenated;
                    // Concatenate duplicates images into one big image
                    std::array<cv::Mat, 2> matrixes{ images[0].image, images[1].image };
                    cv::hconcat(matrixes.data(), 2, concatenated);
                    cv::imshow("Duplicate Finder", concatenated);
                    cv::waitKey(0);
                }
            }
        }
    }

    LOG("[Finished displaying duplicate pairs]");

    cv::destroyAllWindows();
}